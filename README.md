# 🩺 Diabetes Prediction System

A Machine Learning-based web application that predicts the likelihood of diabetes in a patient using clinical input features. The project uses a trained Support Vector Machine (SVM) model along with preprocessing techniques and is deployed using a Streamlit web interface.

---

## 🌐 Live Demo

👉 https://your-streamlit-app-link.streamlit.app

---

## 📌 Project Overview

Diabetes is a chronic disease that affects millions of people worldwide. Early prediction helps in better diagnosis, prevention, and treatment.

This project uses machine learning to classify whether a person is likely to have diabetes based on medical attributes such as:

- Glucose level  
- BMI  
- Age  
- Blood pressure  
- Insulin level  
- Smoking history  
- Other health indicators  

---

## ⚙️ Tech Stack

- Python 🐍  
- Streamlit 🌐  
- Scikit-learn 🤖  
- Pandas 📊  
- NumPy 📊  
- Pickle (Model Serialization)  

---

## 🧠 Machine Learning Model

- **Algorithm used:** Support Vector Machine (SVM)

### Preprocessing:
- StandardScaler for feature scaling  
- Label encoding for categorical variables (gender, smoking history)  

### Model Artifacts:
- `trained_model.sav` → Trained ML model  
- `scaler.sav` → Feature scaler  
- `gender_encoder.sav` → Gender encoder  
- `smoking_encoder.sav` → Smoking encoder  

---

## 📂 Project Structure
DiabetesPrediction/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── trained_model.sav # Trained ML model
├── scaler.sav # Feature scaler
├── gender_encoder.sav # Gender encoder
├── smoking_encoder.sav # Smoking encoder
├── requirements.txt # Dependencies
├── .gitignore
└── README.md

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/arun15112005/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run app.py
🎯 Output
🟢 Low Risk of Diabetes
🔴 High Risk of Diabetes
📌 Future Improvements
Improve model accuracy with hyperparameter tuning
Add probability score for prediction
Add more medical features
Deploy with authentication system
Add data visualization dashboard
👨‍💻 Author

Arun
GitHub: https://github.com/arun15112005

⭐ If you like this project, give it a star!
