#  Import required Python libraries
import pandas as pd                      # For handling tabular data
import numpy as np                       # For numerical operations
import matplotlib.pyplot as plt          # For plotting
import seaborn as sns                    # For pretty plots
import joblib                            # For saving the model

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Read the CSV file into a DataFrame
data = pd.read_csv("college_student_placement_dataset.csv")
print("First few rows of the dataset:")
print(data.head())


# Remove the 'College_ID' column if it exists (not useful for prediction)
data = data.drop(columns=["College_ID"], errors='ignore')

# Remove rows with missing values
data = data.dropna()

# Convert text columns (like 'Yes'/'No') into numeric columns using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Show column names after encoding
print("\nColumns after encoding:", data_encoded.columns.tolist())

# We want to predict 'Placement_Yes'
target_column = [col for col in data_encoded.columns if 'Placement' in col][0]

X = data_encoded.drop(columns=[target_column])  # Features
y = data_encoded[target_column]                 # Target label

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)        # Only transform test data

# We'll use a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# This shows which features contributed the most to the model
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# This checks how stable our model is on different splits of the data
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Average CV Accuracy:", np.mean(cv_scores))


joblib.dump(model, 'placement_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully!")

# Create a DataFrame with one new student's information
new_student = pd.DataFrame([{
    "IQ": 120,
    "Prev_Sem_Result": 75,
    "CGPA": 8.2,
    "Academic_Performance": 85,
    "Extra_Curricular_Score": 60,
    "Communication_Skills": 80,
    "Projects_Completed": 3,
    "Internship_Experience_Yes": 1  # Match the encoded column name
}])

# Reorder columns to match training data and fill missing with 0
new_student = new_student.reindex(columns=X.columns, fill_value=0)

# Scale the input like training data
new_student_scaled = scaler.transform(new_student)

# Predict using the trained model
new_prediction = model.predict(new_student_scaled)
print("\nNew Student Placement Prediction:", "Placed" if new_prediction[0] == 1 else "Not Placed")
