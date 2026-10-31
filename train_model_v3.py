"""
RetinaSense v3 - ordinal-regression DR grader with patient-grouped validation.

Expects a manifest CSV (from prep_data.py) with at least:
  path   - image file path (already cropped / resized / Ben Graham preprocessed)
  label  - integer 0..4
Optional columns:
  source     - dataset name (used by --holdout-source)
  patient_id - used for grouped splitting. If missing, it is derived from the
               file name (EyePACS "10_left.jpeg" -> patient "10").

Images are fed as RAW 0-255 pixels. EfficientNet rescales internally, so do NOT divide by 255.

Example:
  python train_model_v3.py --manifest /kaggle/working/data/manifest.csv \
      --holdout-source idrid --epochs-head 4 --epochs-ft 25
"""
import argparse, json, os, re, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


# ----------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=4)
    p.add_argument("--epochs-ft", type=int, default=25)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=5e-5)
    p.add_argument("--holdout-source", default=None, help="source kept out as external test set")
    p.add_argument("--path-col", default="path")
    p.add_argument("--label-col", default="label")
    p.add_argument("--source-col", default="source")
    p.add_argument("--patient-col", default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0, help="which of 5 grouped folds is validation (0-4)")
    return p.parse_args()


# ----------------------------------------------------------------- data
def derive_patient(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    m = re.match(r"^(\d+)_(left|right)", stem, re.I)
    return m.group(1) if m else stem


def load_manifest(a):
    df = pd.read_csv(a.manifest)
    df = df.rename(columns={a.path_col: "path", a.label_col: "label"})
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(0, 4)].reset_index(drop=True)
    df["source"] = df[a.source_col] if a.source_col in df.columns else "all"
    if a.patient_col in df.columns:
        df["patient"] = df[a.patient_col].astype(str)
    else:
        print("[warn] no patient column found, deriving patient id from file names.")
        df["patient"] = df["path"].map(derive_patient)
    # prefix with source so ids from different datasets never collide
    df["group"] = df["source"].astype(str) + "_" + df["patient"]
    return df


def split(df, a):
    test_df = None
    if a.holdout_source:
        mask = df["source"] == a.holdout_source
        if mask.sum() == 0:
            raise SystemExit(f"holdout source '{a.holdout_source}' not in manifest: {df['source'].unique()}")
        test_df, df = df[mask], df[~mask].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
    folds = list(sgkf.split(df, df["label"], df["group"]))
    tr_idx, va_idx = folds[a.fold]
    tr, va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    assert not set(tr["group"]) & set(va["group"]), "patient leakage between train and val!"
    return tr, va, test_df


def make_loader(size):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [size, size])
        return tf.cast(img, tf.float32), tf.cast(label, tf.float32)  # raw 0-255
    return load


def train_ds(df, a):
    load = make_loader(a.size)
    per_class = []
    for c in sorted(df["label"].unique()):
        sub = df[df["label"] == c]
        d = tf.data.Dataset.from_tensor_slices((sub["path"].values, sub["label"].values.astype("float32")))
        per_class.append(d.shuffle(len(sub), seed=a.seed).repeat())
    ds = tf.data.Dataset.sample_from_datasets(per_class, weights=[1.0 / len(per_class)] * len(per_class), seed=a.seed)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(tf.data.AUTOTUNE)
    steps = max(1, len(df) // a.batch)
    return ds, steps


def eval_ds(df, a):
    load = make_loader(a.size)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype("float32")))
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------- model
def build_model(size):
    from tensorflow.keras import layers
    inp = layers.Input((size, size, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inp)
    x = layers.RandomRotation(0.1, fill_mode="constant")(x)
    x = layers.RandomZoom(0.1, fill_mode="constant")(x)
    x = layers.RandomContrast(0.15)(x)
    base = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False
    x = base(x, training=False)  # keeps BatchNorm in inference mode always
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, name="severity")(x)  # single ordinal score ~0..4
    return tf.keras.Model(inp, out), base


def unfreeze(base):
    base.trainable = True
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False


# ----------------------------------------------------------------- metrics
def to_class(p, th):
    return np.digitize(p, th)


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def tune_thresholds(preds, y, th=(0.5, 1.5, 2.5, 3.5)):
    th = list(th)
    best = qwk(y, to_class(preds, th))
    for _ in range(3):
        for i in range(4):
            for cand in np.linspace(th[i] - 0.5, th[i] + 0.5, 41):
                t = th.copy(); t[i] = cand
                if t != sorted(t):
                    continue
                s = qwk(y, to_class(preds, t))
                if s > best:
                    best, th = s, t
    return th, best


def report(name, y, pred):
    cm = confusion_matrix(y, pred, labels=range(5))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    ref_t, ref_p = y >= 2, pred >= 2       # referable DR = Moderate or worse
    dr_t, dr_p = y >= 1, pred >= 1         # any DR
    def bin_stats(t, p):
        tp, tn = (t & p).sum(), (~t & ~p).sum()
        return dict(acc=float((p == t).mean()),
                    sensitivity=float(tp / max(t.sum(), 1)),
                    specificity=float(tn / max((~t).sum(), 1)))
    m = dict(n=int(len(y)),
             accuracy_5class=float((pred == y).mean()),
             within_one=float((np.abs(pred - y) <= 1).mean()),
             qwk=float(qwk(y, pred)),
             binary_referable=bin_stats(ref_t, ref_p),
             binary_any_dr=bin_stats(dr_t, dr_p),
             per_class_recall={CLASSES[i]: float(recall[i]) for i in range(5)},
             pred_distribution={CLASSES[i]: int((pred == i).sum()) for i in range(5)},
             confusion_matrix=cm.tolist())
    print(f"\n===== {name} (n={m['n']}) =====")
    print(f"5-class acc {m['accuracy_5class']:.3f} | within-one {m['within_one']:.3f} | QWK {m['qwk']:.3f}")
    r = m["binary_referable"]
    print(f"Referable DR: acc {r['acc']:.3f} sens {r['sensitivity']:.3f} spec {r['specificity']:.3f}")
    print("Per-class recall:", {k: round(v, 3) for k, v in m["per_class_recall"].items()})
    print("Pred distribution:", m["pred_distribution"])
    print("Confusion matrix (rows=true):\n", cm)
    if max(m["pred_distribution"].values()) > 0.8 * len(y):
        print("[WARN] >80% of predictions fall in one class - possible collapse.")
    return m


def predict_tta(model, ds):
    outs = []
    for f in (lambda x: x,
              lambda x: tf.image.flip_left_right(x),
              lambda x: tf.image.flip_up_down(x)):
        outs.append(model.predict(ds.map(lambda x, y: f(x)), verbose=0).ravel())
    return np.mean(outs, axis=0)


class QWKCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, y_val, patience=7):
        super().__init__()
        self.val_ds, self.y, self.patience = val_ds, y_val, patience
        self.best, self.best_w, self.wait = -1.0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.val_ds, verbose=0).ravel()
        cls = to_class(p, [0.5, 1.5, 2.5, 3.5])
        k = qwk(self.y, cls)
        dist = np.bincount(cls, minlength=5).tolist()
        print(f"  val QWK {k:.4f} | pred dist {dist}")
        if k > self.best:
            self.best, self.best_w, self.wait = k, self.model.get_weights(), 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("  early stopping")
                self.model.stop_training = True


# ----------------------------------------------------------------- main
def main():
    a = get_args()
    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    df = load_manifest(a)
    tr, va, te = split(df, a)
    print(f"train {len(tr)} | val {len(va)} | external test {0 if te is None else len(te)}")
    print("train label counts:", tr["label"].value_counts().sort_index().to_dict())

    tds, steps = train_ds(tr, a)
    vds = eval_ds(va, a)
    y_val = va["label"].values

    model, base = build_model(a.size)
    loss = tf.keras.losses.Huber(delta=1.0)

    print("\n--- Phase 1: head only ---")
    model.compile(tf.keras.optimizers.Adam(a.lr_head), loss=loss)
    model.fit(tds, steps_per_epoch=steps, epochs=a.epochs_head, verbose=2)

    print("\n--- Phase 2: fine-tune (BatchNorm frozen) ---")
    unfreeze(base)
    model.compile(tf.keras.optimizers.Adam(a.lr_ft), loss=loss)
    ckpt = QWKCheckpoint(vds, y_val)
    cbs = [ckpt, tf.keras.callbacks.ReduceLROnPlateau("loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)]
    model.fit(tds, steps_per_epoch=steps, epochs=a.epochs_ft, callbacks=cbs, verbose=2)
    if ckpt.best_w is not None:
        model.set_weights(ckpt.best_w)

    # thresholds tuned on validation only
    val_pred = predict_tta(model, vds)
    th, tuned = tune_thresholds(val_pred, y_val)
    print(f"\nTuned thresholds: {np.round(th, 3).tolist()} (val QWK {tuned:.4f})")

    results = {"thresholds": th, "validation": report("VALIDATION (patient-grouped)", y_val, to_class(val_pred, th))}
    if te is not None:
        te_pred = predict_tta(model, eval_ds(te, a))
        results["external_test"] = report(f"EXTERNAL TEST [{a.holdout_source}]", te["label"].values, to_class(te_pred, th))

    # ---- export
    model.save(os.path.join(a.out, "dr_model.keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(os.path.join(a.out, "dr_model.tflite"), "wb") as f:
        f.write(conv.convert())
    meta = dict(task="ordinal_regression", input_size=a.size, input_range="raw 0-255 (no /255)",
                output="single float severity score; class = np.digitize(score, thresholds)",
                thresholds=th, classes=CLASSES)
    json.dump(meta, open(os.path.join(a.out, "model_metadata.json"), "w"), indent=2)
    json.dump(results, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(f"\nSaved to {a.out}/: dr_model.keras, dr_model.tflite, model_metadata.json, metrics.json")


if __name__ == "__main__":
    main()
