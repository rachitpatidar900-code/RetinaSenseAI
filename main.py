"""RetinaSense API - serves the v3 ordinal-regression TFLite model.
Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Put dr_model.tflite and model_metadata.json in ./models/
"""
import json, os
import cv2, numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = os.path.dirname(__file__)
META = json.load(open(os.path.join(BASE, "models", "model_metadata.json")))
SIZE, TH, CLASSES = META["input_size"], META["thresholds"], META["classes"]

interp = tf.lite.Interpreter(model_path=os.path.join(BASE, "models", "dr_model.tflite"))
interp.allocate_tensors()
IN, OUT = interp.get_input_details()[0], interp.get_output_details()[0]

# Set from APTOS training images only: 5th percentile of sharpness at 300x300.
# Replace 0.0 with your p5 value before testing. At 0.0 the sharpness check passes everything.
SHARP_MIN = 46.6

ADVICE = [
    "No signs of diabetic retinopathy detected. Continue routine annual screening.",
    "Mild changes suggested. Re-screen in 6-12 months and keep blood sugar controlled.",
    "Moderate changes suggested. Refer to an eye specialist.",
    "Severe changes suggested. Urgent referral to an ophthalmologist.",
    "Proliferative changes suggested. Urgent referral to an ophthalmologist.",
]
DISCLAIMER = "Screening-support prototype only. Not a diagnosis. A qualified clinician must review the result."

app = FastAPI(title="RetinaSense")


# ---- MUST match training preprocessing (prep_data.py). Edit here if yours differs. ----
def crop_black_border(img, tol=7):
    mask = img.max(axis=2) > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def prep_base(bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = crop_black_border(img)
    return cv2.resize(img, (SIZE, SIZE))  # uint8 RGB, before Ben Graham


def preprocess(bgr):
    img = prep_base(bgr)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), SIZE / 30), -4, 128)  # Ben Graham
    return img.astype(np.float32)  # raw 0-255, NO /255
# ---------------------------------------------------------------------------------------


def quality(bgr_original, base_rgb):
    # Measured on the same image the SHARP_MIN percentile was computed from (before Ben Graham).
    h, w = bgr_original.shape[:2]
    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = float(gray.mean())
    ok = sharp > SHARP_MIN and 25 < bright < 220 and min(h, w) >= 200
    return dict(ok=bool(ok), sharpness=round(float(sharp), 1),
                brightness=round(bright, 1), original_size=[h, w])


def predict_score(x):
    # average over flips (same TTA used in training evaluation)
    scores = []
    for v in (x, x[:, ::-1], x[::-1]):
        interp.set_tensor(IN["index"], np.ascontiguousarray(v)[None].astype(IN["dtype"]))
        interp.invoke()
        scores.append(float(interp.get_tensor(OUT["index"]).ravel()[0]))
    return float(np.mean(scores)), float(np.std(scores))


@app.get("/health")
def health():
    return {"status": "ok", "task": META["task"], "thresholds": TH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not read image. Upload a JPG or PNG fundus photo.")
    base = prep_base(bgr)                  # before Ben Graham, used for quality check
    model_input = preprocess(bgr)          # full training preprocessing, used for the model
    q = quality(bgr, base)
    score, spread = predict_score(model_input)
    cls = int(np.digitize(score, TH))
    edges = [-1e9] + TH + [1e9]
    margin = min(score - edges[cls], edges[cls + 1] - score)  # distance to nearest cut-point
    confidence = "high" if margin > 0.35 and spread < 0.3 else "medium" if margin > 0.15 else "low"
    if not q["ok"]:
        confidence = "low"  # poor-quality images can't support a high-confidence result
    return {
        "grade": cls, "label": CLASSES[cls], "score": round(score, 3),
        "confidence": confidence, "referable": cls >= 2,
        "advice": ADVICE[cls], "quality": q,
        "quality_warning": None if q["ok"] else "Image looks blurry, dark or small. Retake for a reliable result.",
        "disclaimer": DISCLAIMER,
    }

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")