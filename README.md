# ⚡ Electricity Theft Detection using Deep Neural Network (DNN)

## 📌 Overview

Electricity theft is a major challenge for power distribution companies, leading to significant financial losses and system inefficiencies. This project presents a **machine learning-based solution** that leverages a **Deep Neural Network (DNN)** to identify suspicious electricity consumption patterns and detect potential theft cases.

The model is trained on historical electricity usage data and deployed using a **Streamlit web application** for real-time predictions.

---

## 🎯 Objectives

* Detect abnormal electricity consumption patterns
* Classify consumers as **Normal (0)** or **Suspicious/Theft (1)**
* Build a scalable and deployable ML pipeline
* Provide an interactive interface for predictions

---

## 🧠 Key Features

* ✔️ End-to-end ML pipeline implementation
* ✔️ Missing value handling using **Median Imputation**
* ✔️ Feature scaling using **StandardScaler**
* ✔️ Class imbalance handling using **SMOTE**
* ✔️ Deep Neural Network (DNN) for classification
* ✔️ Real-time prediction via **Streamlit app**
* ✔️ CSV-based batch prediction support

---

## 🛠️ Tech Stack

| Category           | Tools / Libraries        |
| ------------------ | ------------------------ |
| Programming        | Python                   |
| Data Handling      | Pandas, NumPy            |
| ML Preprocessing   | Scikit-learn             |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Deep Learning      | TensorFlow / Keras       |
| Deployment         | Streamlit                |
| Version Control    | Git & GitHub             |

---

## 📂 Project Structure

```
Electricity_Theft_Detection/
│
├── app.py                       # Streamlit web app
├── requirements.txt            # Dependencies
├── electricity_theft_dnn.keras # Trained DNN model
├── imputer.pkl                 # Missing value handler
├── scaler.pkl                  # Feature scaler
├── feature_columns.pkl         # Feature order
├── sample_input.csv            # Example input file
└── README.md                   # Project documentation
```

---

## ⚙️ ML Pipeline

1. Data Loading
2. Data Cleaning & Preprocessing
3. Handling Missing Values (Median Imputation)
4. Feature Scaling (Standardization)
5. Train-Test Split
6. Class Imbalance Handling (SMOTE)
7. Model Training (DNN)
8. Model Evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)
9. Model Saving (for deployment)

---

## 🧮 Model Architecture

* Input Layer (based on feature size)
* Dense Layer (128 neurons, ReLU)
* Dropout (0.3)
* Dense Layer (64 neurons, ReLU)
* Dropout (0.2)
* Dense Layer (32 neurons, ReLU)
* Output Layer (1 neuron, Sigmoid)

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

> ⚠️ Note: Special focus is given to **Recall**, as detecting theft cases is critical.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/electricity-theft-detection.git
cd electricity-theft-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📥 Input Format

* Upload a CSV file with electricity consumption data
* Must follow the same structure as training dataset

---

## 📤 Output

The application provides:

* **Theft Probability (0 to 1)**
* **Prediction Label**

  * `0` → Normal Consumer
  * `1` → Suspicious / Possible Theft

---

## 🌐 Deployment

This project is deployed using **Streamlit**, enabling:

* Easy access via browser
* Real-time prediction
* User-friendly interface

---

## 🔍 Future Enhancements

* Compare with advanced models (Random Forest, XGBoost)
* Implement time-series models (LSTM)
* Add explainability (SHAP, LIME)
* Integrate with smart meter systems
* Build interactive dashboards (Power BI / Plotly)

---

## 👨‍💻 Author

**Jatin Singh**
B.Tech CSE (AI & ML)
Aspiring Machine Learning Engineer



