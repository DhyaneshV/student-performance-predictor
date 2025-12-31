# 🎓 Student Performance Predictor

A machine learning project that predicts whether a student will **pass or fail** based on academic and personal factors.  
This project demonstrates a complete ML workflow using real-world data.

---

## 📌 Project Overview
The model is trained on a student performance dataset and predicts outcomes using a **Logistic Regression** classifier.  
The trained model is saved and later used in a simple **Flask web application** for real-time predictions.

---

## 🧠 Features Used
- Study time
- Past failures
- Absences
- Health condition
- Other academic and demographic attributes

---

## ⚙️ Tech Stack
- **Python**
- **Pandas**
- **scikit-learn**
- **Flask**
- **Joblib**

---

## 🚀 How It Works
1. Dataset is loaded and preprocessed
2. Categorical features are encoded
3. Model is trained using Logistic Regression
4. Model accuracy is evaluated
5. Trained model is saved (`student_model.pkl`)
6. Flask app loads the model and predicts results from user input

---

## 🏃‍♂️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
