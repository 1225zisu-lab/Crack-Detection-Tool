# Crack Detection & Measurement Tool (MATLAB)
**Developer: Om Ray**

A MATLAB-based application for detecting and measuring structural cracks from images using Machine Learning and classical image processing techniques.  
This tool supports automated detection, crack length estimation, batch testing, and PDF report generation.

---

## ✅ Key Features
✅ SVM-based crack classification  
✅ Feature extraction (GLCM, LBP, Hu Moments)  
✅ Automatic crack segmentation + length estimation  
✅ Calibration (mm/pixel) support  
✅ Manual 2-point measurement  
✅ GUI interface (no coding needed)  
✅ PDF reporting (includes results + annotated image)  
✅ Batch processing of folder images  
✅ Annotated image export  

---

## 📁 Project Structure
Crack Detection/
│
├── code/
│ ├── crackDetectionApp_v2.m → Main GUI
│ ├── train_model.m → SVM training script
│ ├── extractFeatures.m → Feature extractor
│ ├── computeHuMoments.m → Hu moment util
│
├── results/
│ └── models/
│ └── SVM_crack_detector_v1.mat → Trained model
│
├── data/
│ └── crack_dataset/ → (not included)
│ ├── crack/
│ └── no_crack/
│
├── reports/ → PDF results (generated)
└── README.md
YAML FILE

> ⚠️ Datasets are NOT uploaded to GitHub.  
> Place raw images under `data/crack_dataset/crack/` and `no_crack/`.

---

## 🚀 Getting Started

### ✅ **1) Requirements**
- MATLAB R2021a or newer
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

---

### ✅ **2) Setup**
Clone this repository:

```bash
git clone https://github.com/1225zisu-lab/Crack-Detection-Tool.git


Open MATLAB → Add project folder to path:

addpath(genpath(pwd))

Open MATLAB → Add project folder to path:

✅ Training the SVM Model (Optional)

If you want to retrain / use your own dataset:

1️⃣ Place images:
data/crack_dataset/crack/
data/crack_dataset/no_crack/

Run in MATLAB:

train_model


This generates:

results/models/SVM_crack_detector_v1.mat



Running the GUI Tool
crackDetectionApp_v2



GUI Actions
Action	         Description
Load Image	     Load new test image
Detect	         Predict crack vs no-crack
Auto Mask	     Generates crack mask & measures length
Calibrate	     Define mm/pixel scaling
Manual Measure 	 Measure manually by 2 points
Export PDF	     Save report
Save Annotated	 Save processed image
Batch Test	     Evaluate folder images


📄 PDF Report Contents

Original input image

Detection label + confidence

Estimated crack length

Approx width (optional)

Timestamp + metadata

Annotated crack visualization

Saved under:

/reports/Test_Results_DD-MM-YYYY.pdf
