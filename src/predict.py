import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/disease_model.pkl")

# New patient data
patient = pd.DataFrame({
    "Age": [45],
    "Gender": [1],
    "BP": [140],
    "Cholesterol": [220],
    "BMI": [28],
    "Glucose": [120],
    "HeartRate": [85],
    "Smoking": [1]
})

# Predict
prediction = model.predict(patient)

if prediction[0] == 1:
    print("Disease Detected")
else:
    print("No Disease Detected")
