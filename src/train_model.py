import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_data.csv")

# Features (Input)
X = df.drop("Disease", axis=1)

# Target (Output)
y = df["Disease"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "models/disease_model.pkl")

print("Model Saved Successfully")