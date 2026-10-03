# ANN-Based Customer Churn Prediction

An end-to-end **Artificial Neural Network (ANN)** project for predicting customer churn using demographic, financial, and account-related features. The trained model is integrated into an interactive **Streamlit** application and deployed for real-time inference.

## 🔗 Live Demo

**[Launch the Customer Churn Prediction App](https://ann-classification-msga88z4jmnzvharlhnwbf.streamlit.app/)**

## 📌 Overview

Customer churn prediction is a binary classification problem in which the objective is to identify customers who are likely to discontinue a service.

This project implements a neural-network-based solution that:

* Preprocesses categorical and numerical features
* Encodes categorical variables using `OneHotEncoder` and `LabelEncoder`
* Scales numerical features
* Trains an Artificial Neural Network using TensorFlow/Keras
* Saves the trained model and preprocessing artifacts
* Performs real-time predictions through Streamlit
* Deploys the application for online inference

## 🏗️ Solution Architecture

```text
                    Customer Data
                         │
                         ▼
                ┌─────────────────┐
                │ Data Preprocess. │
                └────────┬────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Categorical Features     Numerical Features
             │                       │
             ▼                       ▼
       Encoding                  Scaling
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌─────────────────┐
                │      ANN Model   │
                │   TensorFlow     │
                │     / Keras      │
                └────────┬────────┘
                         │
                         ▼
                 Churn Probability
                         │
                         ▼
                 Streamlit Web App
                         │
                         ▼
                  Prediction Result
```

## 🧠 Model

The project uses an **Artificial Neural Network** for binary classification.

The model receives processed customer attributes and produces a probability representing the likelihood of customer churn.

### Input Features

| Feature            | Description                      |
| ------------------ | -------------------------------- |
| Geography          | Customer's geographical location |
| Gender             | Customer gender                  |
| Age                | Customer age                     |
| Credit Score       | Customer credit score            |
| Balance            | Customer account balance         |
| Tenure             | Customer relationship duration   |
| Number of Products | Number of banking products       |
| Has Credit Card    | Credit card ownership            |
| Is Active Member   | Customer activity status         |
| Estimated Salary   | Estimated customer salary        |

### Target

```text
Exited
0 → Customer retained
1 → Customer churned
```

## ⚙️ Technologies

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

## 🔄 Data Preprocessing

The application uses the same preprocessing pipeline during inference that was used during model development.

### Categorical Encoding

**Geography**

```python
OneHotEncoder
```

**Gender**

```python
LabelEncoder
```

### Numerical Scaling

Numerical features are transformed using a trained scaler before being passed to the neural network.

The preprocessing artifacts are persisted and loaded by the Streamlit application.

## 📁 Repository Structure

```text
ANN-classification/
│
├── app.py
├── Churn_Modelling.csv
│
├── experiments.ipynb
├── prediction.ipynb
│
├── model.h5
├── scaler.pkl
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
│
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.10 / 3.11
* Git
* pip

### 1. Clone the repository

```bash
git clone https://github.com/bharat-02/ANN-classification.git
cd ANN-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows CMD**

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

**[Open the deployed application →](https://ann-classification-msga88z4jmnzvharlhnwbf.streamlit.app/)**

The deployed application provides an interactive interface where users can enter customer attributes and receive a churn prediction.

## 📊 Prediction Workflow

```text
User Input
    ↓
Input Validation
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
ANN Inference
    ↓
Churn Probability
    ↓
Classification Result
```

## 💾 Model Artifacts

The repository contains the trained model and preprocessing artifacts required for inference:

| File                       | Purpose                  |
| -------------------------- | ------------------------ |
| `model.h5`                 | Trained ANN model        |
| `scaler.pkl`               | Numerical feature scaler |
| `onehot_encoder_geo.pkl`   | Geography encoder        |
| `label_encoder_gender.pkl` | Gender encoder           |

Persisting these artifacts ensures that inference uses the same transformations established during model development.

## 📓 Development Notebooks

### `experiments.ipynb`

Contains the model development and experimentation workflow, including data preparation and ANN training.

### `prediction.ipynb`

Contains the prediction and inference workflow for the trained model.

## 🔬 Key Implementation Concepts

This project demonstrates practical implementation of:

* Binary classification
* Artificial Neural Networks
* Feature engineering
* One-hot encoding
* Label encoding
* Feature scaling
* Model persistence
* Real-time model inference
* Streamlit application development
* Machine learning deployment

## 🔮 Future Enhancements

Potential improvements include:

* Hyperparameter optimization
* Cross-validation
* Model performance benchmarking
* Explainable AI integration
* Prediction probability visualization
* Improved input validation
* Model versioning
* Automated CI/CD deployment
* Production-grade API integration

## 👨‍💻 Author

### Bharat Kumar

**Data Science & Machine Learning Enthusiast**

* GitHub: [bharat-02](https://github.com/bharat-02)
* LinkedIn: [Bharat Kumar](https://www.linkedin.com/in/bharat-kumar-a74b26346/)

---

⭐ If you find this project useful, consider giving the repository a star.

**Live Demo:** [ANN Customer Churn Prediction](https://ann-classification-msga88z4jmnzvharlhnwbf.streamlit.app/)
