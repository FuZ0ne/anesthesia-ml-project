"""Streamlit interface for the anesthesia outcome prediction model."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Anesthesia Outcome Prediction",
    page_icon="🩺",
    layout="centered",
)

# Paths and model loading
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "anesthesia_rf_model.pkl"
@st.cache_resource
def load_model():
    """Load and cache the trained machine-learning model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()

# Application header
st.title("🩺 Anesthesia Outcome Prediction")
st.markdown(
    """
    This application demonstrates a machine-learning model for
    predicting the anesthesia-related outcome from selected
    preoperative and procedural features.
    """
)
st.warning(
    "Research / educational demonstration only. "
    "This model is not intended for clinical decision-making."
)

# Input section
st.subheader("Patient and Procedure Information")
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=50,
    step=1,
)
bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1,
)
gender = st.selectbox(
    "Gender",
    options=["M", "F"],
)
surgery_type = st.selectbox(
    "Surgery Type",
    options=[
        "Cosmetic",
        "Orthopedic",
        "Cardiovascular",
        "Neurological",
    ],
)
surgery_duration = st.number_input(
    "Surgery Duration (minutes)",
    min_value=30,
    max_value=600,
    value=150,
    step=1,
)
anesthesia_type = st.selectbox(
    "Anesthesia Type",
    options=["Local", "General"],
)
has_comorbidities = st.selectbox(
    "Has Comorbidities",
    options=[0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

# Prediction
if st.button(
    "Predict Outcome",
    type="primary",
    use_container_width=True,
):
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "BMI": [bmi],
            "SurgeryType": [surgery_type],
            "AnesthesiaType": [anesthesia_type],
            "SurgeryDuration_min": [surgery_duration],
            "Has_Comorbidities": [has_comorbidities],
        }
    )
    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])
    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)
        st.stop()

    # Results
    st.divider()
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Predicted Outcome: 1")
    else:
        st.success("Predicted Outcome: 0")
    
    st.metric(
        label="Probability of Outcome = 1",
        value=f"{probability:.1%}",
    )
    st.progress(
        min(max(probability, 0.0), 1.0)
    )

# Footer
st.divider()
st.caption(
    "Educational machine-learning demonstration. "
    "Predictions should not be used as a substitute for clinical judgment."
)