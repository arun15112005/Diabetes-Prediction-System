import streamlit as st
import pickle
import numpy as np

# Load saved files
model = pickle.load(open("trained_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
gender_encoder = pickle.load(open("gender_encoder.sav", "rb"))
smoking_encoder = pickle.load(open("smoking_encoder.sav", "rb"))

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Risk Prediction System")
st.markdown(
    "Enter patient health information below to estimate diabetes risk."
)

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:
    gender = st.radio(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

with col2:
    heart_disease = st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400,
        value=100
    )

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever",
            "No Info"
        ]
    )

st.divider()

# Convert Yes/No to 1/0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

# Prediction Button
if st.button("🔍 Predict Diabetes Risk"):

    gender_encoded = gender_encoder.transform([gender])[0]
    smoking_encoded = smoking_encoder.transform([smoking_history])[0]

    input_data = np.array([[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        smoking_encoded,
        bmi,
        hba1c,
        blood_glucose
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(
            "⚠️ High Risk of Diabetes\n\n"
            "The model predicts that this patient may have diabetes. "
            "Please consult a healthcare professional for proper medical evaluation."
        )
    else:
        st.success(
            "✅ Low Risk of Diabetes\n\n"
            "The model predicts that this patient is unlikely to have diabetes."
        )

st.divider()

st.caption(
    "This prediction is generated using a Machine Learning model and should not be considered medical advice."
)