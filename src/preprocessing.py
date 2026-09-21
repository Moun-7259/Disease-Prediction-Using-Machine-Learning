import pandas as pd

# Load dataset
df = pd.read_csv("dataset/disease_data.csv")

# Display first 5 rows
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Save cleaned data
df.to_csv("outputs/cleaned_data.csv", index=False)

print("\nData Cleaning Completed")
