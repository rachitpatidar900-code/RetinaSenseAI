# RetinaSense AI

A research prototype that grades diabetic retinopathy (DR) from retinal fundus photos.

**This is not a diagnostic tool and is not ready for screening.** Do not use it to make decisions about patients. A qualified clinician must review any result.

## What it does

You upload a fundus photo. The app returns:

- a DR grade from 0 (No DR) to 4 (Proliferative DR)
- a score and a confidence label
- a referable flag (Moderate or worse)

If the image fails the quality check, the grade is hidden and you are asked to retake the photo.

## Results

**Validation (APTOS 2019, one patient-grouped fold, n=733):** QWK 0.929, 5-class accuracy 0.820, referable sensitivity 0.978, specificity 0.908. Thresholds were tuned on this same fold, so these numbers are optimistic.

**External test (IDRiD disease grading testing set, n=103):** Using the TFLite model with thresholds fixed from APTOS validation:

| Metric | Result |
|---|---|
| 5-class accuracy | 0.476 |
| Within one grade | 0.816 |
| Quadratic weighted kappa | 0.624 |
| Referable sensitivity | 0.812 |
| Referable specificity | 0.897 |

Mild has only 5 images in this set, so its results are very noisy. IDRiD has since been used for development checks, so it is no longer a clean test set.

## Known limitations

- Performance drops sharply on a second dataset. About 1 in 5 referable cases were missed on IDRiD.
- Scores rank grades reasonably (Spearman 0.685), but the cut-points do not transfer well.
- Brightness affects the score. Darkening one sample image moved its score from 2.346 to 2.136.
- Mild and Severe are the weakest grades.
- The model was trained on one dataset. There has been no prospective or clinical validation.

## Quality check

- Sharpness is measured on the 300×300 image before enhancement. The threshold (`SHARP_MIN = 46.6`) is the 5th percentile of APTOS training images at that size. It was not tuned on IDRiD.
- Brightness must be between 25 and 220.
- The original image must be at least 200 pixels on its shorter side.

## Run the app

1. Install **Python 3.12** from python.org. Tick "Add python.exe to PATH". Python 3.13 and later do not work with TensorFlow on Windows.
2. Download this repo as a ZIP and unzip it, or use `git clone`.
3. Double-click `run_app.bat`. The first run installs the packages and takes several minutes. After that, the browser opens to http://localhost:8000 in a few seconds.
4. Close the black window to stop the app.

## Environment

- Tested on Windows with Python 3.12.
- Local TFLite results match the app on a 10-image sample.
- Kaggle results differ from local by up to 0.023 on the same files, with no grade changes in that sample. The cause is probably a different TensorFlow build, but this is not confirmed.

## Data and licences

- **APTOS 2019 Blindness Detection:** Kaggle competition data. Check the competition rules before redistributing trained model weights.
- **IDRiD:** CC BY 4.0. Source: IEEE DataPort, Indian Diabetic Retinopathy Image Dataset. Please cite it if you use it.

## Next steps

A larger and more varied training set (for example EyePACS, subject to its terms) is needed before the model can be evaluated for generalisation. Any new evaluation needs a fresh held-out test set that has not been used for development.

## Disclaimer

Screening-support prototype only. Not a diagnosis. A qualified clinician must review the result.