import glob, os, cv2
from main import preprocess, predict_score

files = sorted(glob.glob("sample_images/**/*.*", recursive=True))
print("Files found:", len(files))

for p in files:
    img = cv2.imread(p)
    if img is None:
        print("could not read", p)
        continue
    s, _ = predict_score(preprocess(img))
    print(os.path.basename(p), round(s, 3))