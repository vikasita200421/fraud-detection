import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Normalize 'Amount' column
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

# Separate features & target
X = df.drop(["Class", "Time"], axis=1)  # Drop 'Time' column
y = df["Class"]

# Balance the dataset using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save preprocessed data as a pickle file (for ML training)
preprocessed_data = {"X": X_resampled, "y": y_resampled}
joblib.dump(preprocessed_data, "preprocessed_data.pkl")

# Convert to DataFrame for Power BI & React.js
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['is_fraud'] = y_resampled  # Add fraud labels

# Save as CSV for Power BI & React.js
df_resampled.to_csv("fraud_data.csv", index=False)

print("✅ Data preprocessing completed!")
print("📁 Saved as 'preprocessed_data.pkl' for ML training.")
print("📊 Saved as 'fraud_data.csv' for Power BI & React.js.")
