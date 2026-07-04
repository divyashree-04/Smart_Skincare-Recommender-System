# How to Train the Real AI Model

## Step 1 — Install requirements
```
pip install -r requirements.txt
```

## Step 2 — Get a dataset

Download from Kaggle:
https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset

Or search: "skin type classification dataset" on Kaggle.

## Step 3 — Organize into this exact folder structure

```
skincare_app/
└── dataset/
    ├── train/
    │   ├── Oily/          ← 160-200 face images
    │   ├── Dry/           ← 160-200 face images
    │   ├── Combination/   ← 160-200 face images
    │   ├── Sensitive/     ← 160-200 face images
    │   └── Normal/        ← 160-200 face images
    └── val/
        ├── Oily/          ← 40-50 face images
        ├── Dry/           ← 40-50 face images
        ├── Combination/   ← 40-50 face images
        ├── Sensitive/     ← 40-50 face images
        └── Normal/        ← 40-50 face images
```

IMPORTANT:
- Folder names must be EXACTLY as above (capital first letter)
- Use JPG or PNG images only
- Images should be clear face photos

## Step 4 — Train

```
python train_model.py
```

Training takes 15-30 minutes on CPU, 5-10 minutes on GPU.

When complete you will see:
```
✅ DONE — Model saved to: skin_model.h5
```

## Step 5 — Run the app

```
python app.py
```

Open browser: http://localhost:5000

Now when a user uploads a photo → real AI prediction runs.
If no photo uploaded (manual path) → manual selection is used.

## What happens if I run app.py without training?

The app still works perfectly.
It shows a message in the console:
  "No model file found — running in demo mode"

Users who take the manual path get real recommendations.
Users who upload a photo get demo predictions (random but realistic).

Once you train and place skin_model.h5 in the folder, real AI kicks in automatically.

## Files generated after training

| File | Description |
|------|-------------|
| skin_model.h5 | Trained model used by the app |
| training_history.png | Accuracy and loss graphs |
| confusion_matrix.png | Shows prediction accuracy per class |
