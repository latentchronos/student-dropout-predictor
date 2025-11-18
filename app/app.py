import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Constants
MODEL_DIR = 'models/'
DATA_DIR = 'data/'

# Paths to the artifacts
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Loading Artifacts
# st.cache_resource is best for efficient loading
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        st.error(
            "ERROR: Model artifacts not found."
            "Please run the '1_EDA_and_Preprocessing.ipynb' and '2_Model_Training.ipynb' notebooks first."
        )
        return None, None, None
    return preprocessor, model, label_encoder

preprocessor, model, label_encoder = load_artifacts()
if preprocessor is None:
    st.stop()
    
# Getting the class names
class_names = label_encoder.classes_

# Feature Lists (from Phase 2)
numerical_cols = [
    'Application order', 'Age at enrollment', 'Admission grade',
    'Previous qualification (grade)',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

categorical_cols = [
    'Marital Status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

# Function for getting user inputs
def get_user_inputs():

    st.sidebar.header("Enter Student Data")
    

    inputs = {}
    
    # Demographics & Personal
    st.sidebar.subheader("Demographics")
    inputs['Age at enrollment'] = st.sidebar.slider("Age at Enrollment", 17, 70, 20)
    
    
    gender_map = {0: "Female", 1: "Male"}
    inputs['Gender'] = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: gender_map[x])
    
    # 1 = single, 2 = married, 3 = widower, 4 = divorced, 5 = facto union, 6 = legally separated
    #marital_stat = {1: "single", 2: "married", 3: "widower", 4: "divorced", 5: "facto union", 6: "legally separated"}
    #inputs['Marital Status'] = st.sidebar.selectbox("Marital Status", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: marital_stat[x])

    inputs['Marital Status'] = st.sidebar.number_input("Marital Status (1-6)", min_value=1, max_value=6, value=1)
    
    # Coded value
    inputs['Nacionality'] = st.sidebar.number_input("Nationality (Code)", min_value=1, value=1)
    inputs['International'] = st.sidebar.selectbox("International?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    
    # --- Financial & Support ---
    st.sidebar.subheader("Financial & Support")
    inputs['Debtor'] = st.sidebar.selectbox("Debtor?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    inputs['Tuition fees up to date'] = st.sidebar.selectbox("Tuition Fees Up to Date?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    inputs['Scholarship holder'] = st.sidebar.selectbox("Scholarship Holder?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    inputs['Displaced'] = st.sidebar.selectbox("Displaced?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    inputs['Educational special needs'] = st.sidebar.selectbox("Special Needs?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    
    # --- Application & Previous Education ---
    st.sidebar.subheader("Application & Prior Education")
    inputs['Application mode'] = st.sidebar.number_input("Application Mode (Code)", min_value=1, value=1)
    inputs['Application order'] = st.sidebar.number_input("Application Order (0-6)", min_value=0, max_value=6, value=1)
    inputs['Course'] = st.sidebar.number_input("Course (Code)", min_value=1, value=9147) # Example
    inputs['Previous qualification'] = st.sidebar.number_input("Previous Qualification (Code)", min_value=1, value=1)
    inputs['Previous qualification (grade)'] = st.sidebar.slider("Previous Qualification Grade", 0.0, 200.0, 120.0)
    inputs['Admission grade'] = st.sidebar.slider("Admission Grade", 0.0, 200.0, 120.0)
    
    # 1 = daytime, 0 = evening
    inputs['Daytime/evening attendance'] = st.sidebar.selectbox("Day/Evening", [0, 1], format_func=lambda x: "Evening" if x==0 else "Daytime")

    # --- Parent's Info (Coded) ---
    st.sidebar.subheader("Parental Information (Coded)")
    inputs["Mother's qualification"] = st.sidebar.number_input("Mother's Qualification (Code)", min_value=1, value=1)
    inputs["Father's qualification"] = st.sidebar.number_input("Father's Qualification (Code)", min_value=1, value=1)
    inputs["Mother's occupation"] = st.sidebar.number_input("Mother's Occupation (Code)", min_value=0, value=1)
    inputs["Father's occupation"] = st.sidebar.number_input("Father's Occupation (Code)", min_value=0, value=1)

    # --- Academic Performance (1st Sem) ---
    st.sidebar.subheader("1st Semester Performance")
    inputs['Curricular units 1st sem (credited)'] = st.sidebar.slider("1st Sem: Credited Units", 0, 20, 0)
    inputs['Curricular units 1st sem (enrolled)'] = st.sidebar.slider("1st Sem: Enrolled Units", 0, 26, 6)
    inputs['Curricular units 1st sem (approved)'] = st.sidebar.slider("1st Sem: Approved Units", 0, 26, 5)
    inputs['Curricular units 1st sem (evaluations)'] = st.sidebar.slider("1st Sem: Evaluations", 0, 45, 8)
    inputs['Curricular units 1st sem (grade)'] = st.sidebar.slider("1st Sem: Average Grade", 0.0, 20.0, 12.0)
    inputs['Curricular units 1st sem (without evaluations)'] = st.sidebar.slider("1st Sem: Without Evaluations", 0, 12, 0)

    # --- Academic Performance (2nd Sem) ---
    st.sidebar.subheader("2nd Semester Performance")
    inputs['Curricular units 2nd sem (credited)'] = st.sidebar.slider("2nd Sem: Credited Units", 0, 20, 0)
    inputs['Curricular units 2nd sem (enrolled)'] = st.sidebar.slider("2nd Sem: Enrolled Units", 0, 23, 6)
    inputs['Curricular units 2nd sem (approved)'] = st.sidebar.slider("2nd Sem: Approved Units", 0, 20, 5)
    inputs['Curricular units 2nd sem (evaluations)'] = st.sidebar.slider("2nd Sem: Evaluations", 0, 33, 8)
    inputs['Curricular units 2nd sem (grade)'] = st.sidebar.slider("2nd Sem: Average Grade", 0.0, 20.0, 12.0)
    inputs['Curricular units 2nd sem (without evaluations)'] = st.sidebar.slider("2nd Sem: Without Evaluations", 0, 12, 0)
    
    # --- Macro-economic Factors ---
    st.sidebar.subheader("Macro-economic Factors")
    inputs['Unemployment rate'] = st.sidebar.slider("Unemployment Rate", 7.6, 16.2, 12.4)
    inputs['Inflation rate'] = st.sidebar.slider("Inflation Rate", -0.8, 4.1, 1.2)
    inputs['GDP'] = st.sidebar.slider("GDP", -3.12, 3.51, 0.32)
    
    return inputs

# --- Main Application ---
def main():
    st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
    st.title("🎓 Student Dropout Probability Predictor")
    st.write(
        "This application predicts the academic outcome of a student (Dropout, Enrolled, or Graduate) "
        "using a machine learning model. Please fill in the student's data in the sidebar."
    )

    # Get inputs from the user
    user_inputs = get_user_inputs()

    # Create a "Predict" button
    if st.button("Predict Student Outcome"):
        # 1. Convert user inputs to a DataFrame
        # The DataFrame must have columns in the *exact* order
        # as the preprocessor expects.
        feature_list = numerical_cols + categorical_cols
        
        # Create a dictionary with the correct order
        ordered_inputs = {col: user_inputs[col] for col in feature_list}
        input_df = pd.DataFrame([ordered_inputs])

        # 2. Preprocess the data
        # The preprocessor will scale and one-hot encode the data
        try:
            input_processed = preprocessor.transform(input_df)
            
            # 3. Make prediction (probabilities)
            probabilities = model.predict_proba(input_processed)
            
            # 4. Get the main prediction
            prediction_index = np.argmax(probabilities)
            predicted_class = class_names[prediction_index]

            # 5. Display the results
            st.subheader("Prediction Results")
            
            # Display the main prediction with an icon
            icons = {"Dropout": "⚠️", "Enrolled": "🔄", "Graduate": "✅"}
            st.markdown(
                f"### {icons.get(predicted_class, '📊')} The model predicts the student will **{predicted_class}**"
            )

            # Create a clean DataFrame for the probability chart
            prob_df = pd.DataFrame(probabilities, columns=class_names)
            prob_df = prob_df.T.reset_index()
            prob_df.columns = ["Outcome", "Probability"]

            st.subheader("Prediction Probabilities")
            # Use st.bar_chart to display probabilities
            st.bar_chart(prob_df.set_index("Outcome"))

            with st.expander("Show Raw Input Data"):
                st.write(input_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure all inputs are valid.")

if __name__ == "__main__":
    main()