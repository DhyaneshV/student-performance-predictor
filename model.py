import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("data/student.xlsx")

# Use ONLY these 4 features
X = df[["studytime", "failures", "absences", "health"]]

# Target variable
y = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# Save model (OVERWRITES old model)
joblib.dump(model, "student_model.pkl")
print("New model saved successfully")
