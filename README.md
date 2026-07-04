# Smart Skincare Recommender System

## Overview

The Smart Skincare Recommender System is a Flask-based web application that analyzes facial skin images using a Convolutional Neural Network (CNN) developed with TensorFlow. The system classifies skin conditions and provides personalized skincare recommendations through a simple, interactive, and user-friendly web interface.

---

## Features

- Skin condition classification using Deep Learning
- Personalized skincare recommendations
- Image upload for skin analysis
- CNN model built with TensorFlow
- Flask-based web application
- Displays prediction confidence score
- User-friendly interface

---

## Technologies Used

- Python
- Flask
- TensorFlow
- OpenCV
- NumPy
- Pandas
- HTML
- CSS
- JavaScript

---

## Model Description

This project uses a Convolutional Neural Network (CNN) trained on skin image data to classify different skin conditions. The trained model predicts the skin condition from an uploaded image, and the application generates personalized skincare recommendations based on the prediction.

---

## Dataset

The model was trained using a publicly available skin image dataset obtained from Kaggle. The dataset was used for educational and research purposes.

---

## Project Structure

```text
Smart-Skincare-Recommender-System/
│
├── dataset/
├── screenshots/
├── static/
├── templates/
├── app.py
├── preprocessing.py
├── train_model.py
├── skin_model.h5
├── requirements.txt
├── HOW_TO_TRAIN.md
└── README.md
```

---

## Installation

1. Clone the repository

```bash
git clone <repository-link>
```

2. Install the required dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
python app.py
```

4. Open the application in your web browser.

---

##  Screenshots

Screenshots of the application are available in the **screenshots** folder.

---

##  Future Enhancements

- Deploy the application on cloud platforms such as Render, Railway, or Azure for public access.
- Train the model using a larger and more diverse dataset to improve prediction accuracy and generalization.
- Support additional skin conditions and multiple skin concerns.
- Recommend skincare products based on the detected skin condition.
- Add user authentication and maintain analysis history
- Improve the UI for better user experience

##Author
**Divya Shree V**
