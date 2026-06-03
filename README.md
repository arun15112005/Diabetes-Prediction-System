# 🩺 Diabetes Prediction System

A Machine Learning-based web application that predicts the likelihood of diabetes in a patient using clinical input features. The project uses a trained Support Vector Machine (SVM) model along with preprocessing techniques and is deployed using a Streamlit web interface.

---

## 📌 Project Overview

Diabetes is a chronic disease that affects millions of people worldwide. Early prediction can help in better management and treatment.

This project uses machine learning to classify whether a person is likely to have diabetes based on medical attributes such as:
- Glucose level
- BMI
- Age
- Blood pressure
- Insulin level
- Other health indicators

---

## ⚙️ Tech Stack

- Python 🐍
- Scikit-learn 🤖
- Pandas & NumPy 📊
- Streamlit 🌐
- Joblib / Pickle

---

## 🧠 Machine Learning Model

- Algorithm used: **Support Vector Machine (SVM)**
- Preprocessing:
  - StandardScaler for feature scaling
  - Label encoding for categorical variables (gender, smoking status)
- Model and preprocessors saved as `.sav` files

---

## 📂 Project Structure

```text id="v9m3qp"
DiabetesPrediction/
│
├── app.py                  # Streamlit web app
├── train_model.py          # Model training script
├── trained_model.sav       # Trained ML model
├── scaler.sav              # Feature scaler
├── gender_encoder.sav      # Gender encoder
├── smoking_encoder.sav     # Smoking encoder
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
