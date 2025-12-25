import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#  dataset
df = pd.read_excel("data/student.xlsx")

# 2. Create target variable (Pass / Fail)
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Drop unnecessary columns
df = df.drop(["G3"], axis=1)

# Encode categorical features
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Split features and target
X = df.drop("pass", axis=1)
y = df["pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved successfully.")
