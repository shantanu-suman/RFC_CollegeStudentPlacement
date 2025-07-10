# RFC_CollegeStudentPlacement
# 🎓 College Student Placement Prediction

This project is a machine learning-based web application that predicts whether a college student is likely to be placed based on academic and extracurricular features.

The app is built using:
- Scikit-learn (for model training)
- Streamlit (for user interface)
- Pandas and NumPy (for data processing)


## 📌 Features

- Predict student placement outcome based on:
  - IQ
  - CGPA
  - Semester performance
  - Communication skills
  - Projects completed
  - Internship experience, and more
- Easy-to-use Streamlit web app
- Saves trained model and scaler for future predictions
- Shows feature importance and model evaluation metrics


## 📁 Project Structure
CollegeStudentPlacement
├── app.py # Streamlit app for prediction
├── main.py # Main training script
├── placement_model.pkl # Trained Random Forest model
├── scaler.pkl # StandardScaler instance
├── college_student_placement_dataset.csv # Input dataset
└── README.md # Project documentation
