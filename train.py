import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load preprocessed data
data = joblib.load("preprocessed_data.pkl")
X, y = data["X"], data["y"]

# Debug: Print number of features
print(f"Shape of X (rows, columns): {X.shape}")  # Should print (N, 29) or (N, 4)

# 🔹 Option 1: Use all features (default)
USE_ALL_FEATURES = True  # Set to False if you want to train with only 4 features

# 🔹 Option 2: Use only 4 selected features (modify feature names accordingly)
if not USE_ALL_FEATURES:
    selected_features = ["Feature1", "Feature2", "Feature3", "Feature4"]  # Replace with actual column names
    X = X[selected_features]
    print(f"Using only selected features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Evaluation:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model training completed and saved as 'fraud_model.pkl'!")
