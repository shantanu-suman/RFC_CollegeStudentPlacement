import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("College Student Placement Predictor")

st.write("""
Enter the student's academic and personal details to predict whether they will be placed or not.
""")

# Input fields for the user
iq = st.number_input("IQ", min_value=50, max_value=200, value=100)
prev_sem_result = st.number_input("Previous Semester Result (%)", min_value=0, max_value=100, value=70)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
academic_performance = st.number_input("Academic Performance (%)", min_value=0, max_value=100, value=80)
extra_curricular_score = st.number_input("Extra Curricular Score", min_value=0, max_value=100, value=50)
communication_skills = st.number_input("Communication Skills (%)", min_value=0, max_value=100, value=70)
projects_completed = st.number_input("Projects Completed", min_value=0, max_value=20, value=2)
internship_experience = st.selectbox("Internship Experience", ["No", "Yes"])

# Convert input to DataFrame for prediction
input_data = pd.DataFrame([{
    "IQ": iq,
    "Prev_Sem_Result": prev_sem_result,
    "CGPA": cgpa,
    "Academic_Performance": academic_performance,
    "Extra_Curricular_Score": extra_curricular_score,
    "Communication_Skills": communication_skills,
    "Projects_Completed": projects_completed,
    "Internship_Experience_Yes": 1 if internship_experience == "Yes" else 0
}])

# Reorder columns to match training data and fill any missing columns
expected_columns = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
                    'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed',
                    'Internship_Experience_Yes']
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Placement"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.success("The student is **likely to be placed**.")
    else:
        st.warning("The student is **unlikely to be placed**.")
