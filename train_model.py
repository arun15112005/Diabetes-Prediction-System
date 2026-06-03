import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Convert text columns to numbers
le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
print("Training SVM model...")

model = LinearSVC(max_iter=10000)

model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\nResults")
print("-" * 30)
print("Training Accuracy:", round(train_acc * 100, 2), "%")
print("Testing Accuracy:", round(test_acc * 100, 2), "%")

# Save files
pickle.dump(model, open("trained_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))
pickle.dump(le_gender, open("gender_encoder.sav", "wb"))
pickle.dump(le_smoking, open("smoking_encoder.sav", "wb"))

print("\nModel saved successfully!")
print("Files created:")
print("- trained_model.sav")
print("- scaler.sav")
print("- gender_encoder.sav")
print("- smoking_encoder.sav")