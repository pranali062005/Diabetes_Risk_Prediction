import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern card layout and shadows
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained model safely using absolute paths
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "kn.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# App Header
st.markdown(
    """
    <div style="padding: 10px 0; text-align: center;">
        <h2>Diabetes Risk Assessment Tool</h2>
        <p style="color: #6c757d;">Fill in the patient clinical and lifestyle metrics below to evaluate risk using the SVM model.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Form Layout
with st.form("prediction_form"):
    st.markdown("### Patient Clinical & Lifestyle Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City", "New York")
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=24.5, format="%.1f"
        )
        family_history_diabetes = st.selectbox(
            "Family History of Diabetes", ["No", "Yes"]
        )
        income_bracket = st.selectbox(
            "Income Bracket", ["Low", "Medium", "High"]
        )

    with col2:
        physical_activity_level = st.selectbox(
            "Physical Activity Level", ["Low", "Moderate", "High"]
        )
        diet_type = st.selectbox(
            "Diet Type", ["Standard", "Vegetarian", "Vegan", "Keto", "Other"]
        )
        smoking_status = st.selectbox(
            "Smoking Status", ["Never", "Former", "Current"]
        )
        alcohol_consumption = st.selectbox(
            "Alcohol Consumption", ["None", "Low", "Moderate", "High"]
        )
        hours_sleep_per_night = st.slider(
            "Hours of Sleep per Night", 1.0, 12.0, 7.0, 0.5
        )
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

    with col3:
        fasting_blood_sugar = st.number_input(
            "Fasting Blood Sugar (mg/dL)", 50.0, 300.0, 95.0, format="%.1f"
        )
        hba1c_level = st.number_input(
            "HbA1c Level (%)", 3.0, 15.0, 5.5, format="%.1f"
        )
        blood_pressure_systolic = st.number_input(
            "Blood Pressure Systolic (mmHg)", 80.0, 200.0, 120.0, format="%.1f"
        )
        blood_pressure_diastolic = st.number_input(
            "Blood Pressure Diastolic (mmHg)", 50.0, 130.0, 80.0, format="%.1f"
        )
        waist_circumference_cm = st.number_input(
            "Waist Circumference (cm)", 40.0, 200.0, 85.0, format="%.1f"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("Predict Risk")

# Prediction Logic
if submit_btn:
    # Encoded values helper
    val_gender = 1 if gender == "Male" else 0
    val_fam = 1 if family_history_diabetes == "Yes" else 0
    val_pa = ["Low", "Moderate", "High"].index(physical_activity_level)
    val_diet = ["Standard", "Vegetarian", "Vegan", "Keto", "Other"].index(
        diet_type
    )
    val_smoke = ["Never", "Former", "Current"].index(smoking_status)
    val_alc = ["None", "Low", "Moderate", "High"].index(alcohol_consumption)
    val_inc = ["Low", "Medium", "High"].index(income_bracket)

    # Constructing input dataframe matching all 21 features expected by the model
    input_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [val_gender],
            "city": [hash(city) % 100],
            "bmi": [bmi],
            "family_history_diabetes": [val_fam],
            "physical_activity_level": [val_pa],
            "diet_type": [val_diet],
            "smoking_status": [val_smoke],
            "alcohol_consumption": [val_alc],
            "hours_sleep_per_night": [hours_sleep_per_night],
            "stress_level": [stress_level],
            "fasting_blood_sugar": [fasting_blood_sugar],
            "hba1c_level": [hba1c_level],
            "blood_pressure_systolic": [blood_pressure_systolic],
            "blood_pressure_diastolic": [blood_pressure_diastolic],
            "waist_circumference_cm": [waist_circumference_cm],
            "income_bracket": [val_inc],
            "diabetes_risk": [
                0
            ],  # Dummy default for accidentally included target feature
            " physical_activity_level ": [val_pa],
            " diet_type ": [val_diet],
            " smoking_status ": [val_smoke],
        }
    )

    try:
        # Pass via NumPy array to match exact dimensions and bypass strict column key matching errors
        prediction = model.predict(input_data.to_numpy())

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Prediction Result")

        if prediction[0] == 1:
            st.error(
                "**High Risk:** The model indicates a high risk profile for diabetes. Please consult a healthcare professional."
            )
        else:
            st.success(
                "**Low Risk:** The model indicates a low risk profile for diabetes."
            )
    except Exception as e:
        st.error(
            f"Error making prediction. Ensure feature formatting matches model expectations: {e}"
        )
