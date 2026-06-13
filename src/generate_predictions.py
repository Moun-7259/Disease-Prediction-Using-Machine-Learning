import pandas as pd
import joblib

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_data.csv")

# Load trained model
model = joblib.load("models/disease_model.pkl")

# Features
X = df.drop("Disease", axis=1)

# Generate predictions
df["Prediction"] = model.predict(X)

# Save predictions
df.to_csv("outputs/predictions.csv", index=False)

print("Predictions file created successfully")
