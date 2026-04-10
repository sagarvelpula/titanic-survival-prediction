# Titanic Survival Prediction (CLI + Streamlit App)

This project predicts whether a passenger survived the Titanic disaster using machine learning.
It includes both a **Command-Line Interface (CLI)** version and an **interactive Streamlit web application**.


---

## 🧠 Overview

The model uses passenger features such as:

* Passenger Class (Pclass)
* Gender (Sex)
* Age

to predict survival using a **Decision Tree Classifier**.

The project demonstrates both:

* Core ML logic (CLI-based interaction)
* User-friendly interface (Streamlit web app)

---

## ✨ Features

* Machine learning pipeline with preprocessing
* Hyperparameter tuning using GridSearchCV
* CLI-based prediction system
* Interactive web UI using Streamlit
* Real-time prediction with probability output

---

## 🛠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

---

## Screenshots 

<img width="1920" height="868" alt="Screenshot 2026-04-06 192718" src="https://github.com/user-attachments/assets/93b3d4a1-3b33-46fa-9ee2-1de2fa6cb7ab" />

<img width="1920" height="872" alt="Screenshot 2026-04-06 192732" src="https://github.com/user-attachments/assets/8781ff67-e2ca-4f5e-8b87-98402cd192c1" />

---

## ▶️ How to Run

### 🔹 CLI Version

Run in terminal:

python titanic_predictor.py

Then enter:

* Pclass
* Sex
* Age

---

### 🔹 Streamlit Version

Run locally:

streamlit run app.py

---

## 🌐 Deployment

The Streamlit version is deployed using:

* Streamlit Cloud

---

## 📊 Model Details

* Algorithm: Decision Tree Classifier
* Preprocessing: Imputation + Scaling
* Tuning: GridSearchCV
* Output: Survival prediction with probability

---

## 📈 Example

Input:
Pclass: 1
Sex: female
Age: 25

Output:
Survived ✅

---

## 👨‍💻 Author

Sagar Velpula
