"""Streamlit user interface for diabetes-risk prediction."""

from __future__ import annotations

import streamlit as st

from model_inference import load_pipeline, predict

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Diabetes Risk Prediction App")
st.write(
    "Enter the patient's diagnostic clinical measurements below to evaluate "
    "diabetes risk."
)
st.caption("For educational use only; this result is not a medical diagnosis.")


@st.cache_resource
def get_pipeline():
    """Load the model pipeline once per Streamlit process."""

    return load_pipeline()


try:
    pipeline = get_pipeline()
except Exception as error:
    st.error(f"Error loading the model pipeline: {error}")
    st.stop()

st.subheader("Patient Diagnostic Measurements")
left_column, right_column = st.columns(2)

with left_column:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of times pregnant",
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=120,
        help="Plasma glucose concentration (0 = missing/unknown)",
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=0,
        max_value=200,
        value=70,
        help="Diastolic blood pressure (0 = missing/unknown)",
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness (0 = missing/unknown)",
    )

with right_column:
    insulin = st.number_input(
        "Insulin Level (μU/mL)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-Hour serum insulin (0 = missing/unknown)",
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (0.0 = missing/unknown)",
    )
    pedigree_function = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="Genetic predisposition score",
    )
    age = st.number_input(
        "Age (Years)",
        min_value=1,
        max_value=120,
        value=33,
    )

if st.button("🔍 Predict Risk", width="stretch"):
    patient_values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree_function,
        "Age": age,
    }

    try:
        result = predict(patient_values, pipeline)
    except Exception as error:
        st.error(f"Unable to generate a prediction: {error}")
        st.stop()

    diabetes_percentage = result.diabetes_probability * 100

    st.divider()
    st.subheader("Diagnostic Prediction Result")

    if result.is_diabetic:
        st.error("⚠️ **High Risk of Diabetes**")
        st.write(
            f"The model estimates a **{diabetes_percentage:.1f}% probability** "
            "of diabetes."
        )
    else:
        non_diabetes_percentage = result.non_diabetes_probability * 100
        st.success("✅ **Low Risk / Non-Diabetic**")
        st.write(
            f"The model estimates a **{non_diabetes_percentage:.1f}% probability** "
            "of being non-diabetic."
        )

    st.progress(int(diabetes_percentage))
