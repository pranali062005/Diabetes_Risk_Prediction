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

# Custom CSS for Professional Black & Blue Theme
st.markdown(
    """
    <style>
    /* Multi-shade professional black and dark-blue gradient background */
    .stApp {
        background: linear-gradient(135deg, #090a0f 0%, #111827 50%, #0f172a 100%);
        background-attachment: fixed;
        color: #f3f4f6;
    }

    /* Dark charcoal card containers with soft blue-tinted shadows */
    div[data-testid="stForm"] {
        background-color: #121824;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 20px rgba(37, 99, 235, 0.12);
        border: 1px solid #1e293b;
    }

    /* Text readability tuning for labels and headers inside the form */
    div[data-testid="stForm"] label, 
    div[data-testid="stForm"] h3, 
    div[data-testid="stForm"] p {
        color: #f3f4f6 !important;
    }

    /* Vibrant blue gradient action button with glow */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained model safely with dynamic path resolution
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(base_dir, "SVC.pkl"),
        os.path.join(base_dir, "models", "SVC.pkl"),
        os.path.join(base_dir, "..", "SVC.pkl"),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        files_in_dir = os.listdir(base_dir)
        raise FileNotFoundError(
            f"Could not find 'kn.pkl'. Files found in app directory ({base_dir}): {files_in_dir}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# App Header
st.markdown(
    """
    <div style="padding: 10px 0; text-align: center;">
        <h2 style="color: #f3f4f6;">Diabetes Risk Assessment Tool</h2>
        <p style="color: #94a3b8;">Fill in the patient clinical and lifestyle metrics below to evaluate risk using the SVM model.</p>
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
    val_gender = 1 if gender == "Male" else 0
    val_fam = 1 if family_history_diabetes == "Yes" else 0
    val_pa = ["Low", "Moderate", "High"].index(physical_activity_level)
    val_diet = ["Standard", "Vegetarian", "Vegan", "Keto", "Other"].index(
        diet_type
    )
    val_smoke = ["Never", "Former", "Current"].index(smoking_status)
    val_alc = ["None", "Low", "Moderate", "High"].index(alcohol_consumption)
    val_inc = ["Low", "Medium", "High"].index(income_bracket)

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
            "diabetes_risk": [0],
            " physical_activity_level ": [val_pa],
            " diet_type ": [val_diet],
            " smoking_status ": [val_smoke],
        }
    )

    try:
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
