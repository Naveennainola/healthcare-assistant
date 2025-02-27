import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
dataset_path = "assistant/dataset/cleaned_dataset.csv"  # Update with the correct path
df = pd.read_csv(dataset_path)

# Convert symptoms into a list
df["Symptoms"] = df["Symptoms"].apply(lambda x: x.split(", "))  # Convert symptoms to list format

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])  # Convert symptoms to binary format
y = df["Disease"]

# Split data for training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Ensure directory exists before saving
model_dir = "assistant/saved_models"
os.makedirs(model_dir, exist_ok=True)

# Save model & MultiLabelBinarizer in correct location
joblib.dump(model, os.path.join(model_dir, "disease_model.pkl"))
joblib.dump(mlb, os.path.join(model_dir, "mlb.pkl"))

# ✅ Model Testing & Performance Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Model training complete! Accuracy: {accuracy:.2f}")
print("🔍 Classification Report:\n", report)
print(f"📂 Model saved at: {model_dir}/disease_model.pkl")
