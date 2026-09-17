import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load processed dataset
df = pd.read_csv("data/processed.csv")

print("Processed Dataset:")
print(df)

# Separate features and target
X = df.drop("result", axis=1)
y = df["result"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/model.pkl")

print("Model saved successfully at: model/model.pkl")