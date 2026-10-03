# ANN Classification — Customer Churn Prediction

An **Artificial Neural Network (ANN)** based customer churn prediction project built with **TensorFlow/Keras** and deployed as an interactive **Streamlit** web application.

The model predicts whether a bank customer is likely to **exit/churn** based on customer information such as geography, gender, age, credit score, balance, tenure, number of products, credit card ownership, and active membership.

## 🚀 Live Demo

👉 **[Try the Live Application](https://ann-classification-msga88z4jmnzvharlhnwbf.streamlit.app/)**

## 📌 Project Overview

Customer churn is an important problem for banks and financial institutions. Predicting customers who are likely to leave can help organizations identify at-risk customers and take appropriate retention actions.

This project uses an **Artificial Neural Network (ANN)** to perform binary classification:

* `0` → Customer stays
* `1` → Customer exits

The project includes data preprocessing, model training, model evaluation, prediction, and deployment using Streamlit.

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Encoding
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Artificial Neural Network
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Save Model & Preprocessors
     ↓
Streamlit Application
     ↓
Customer Churn Prediction
```

## 📊 Dataset

The project uses the **Churn Modelling** dataset.

Important input features include:

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| Geography        | Customer's country/region                |
| Gender           | Customer's gender                        |
| Age              | Customer age                             |
| Credit Score     | Customer credit score                    |
| Balance          | Account balance                          |
| Tenure           | Number of years with the bank            |
| Num Of Products  | Number of banking products               |
| Has Credit Card  | Whether the customer has a credit card   |
| Is Active Member | Whether the customer is an active member |
| Estimated Salary | Customer's estimated salary              |

### Target Variable

**Exited**

```text
0 → Customer did not leave
1 → Customer exited
```

## 🤖 Artificial Neural Network

The project uses an ANN for binary classification.

The basic architecture consists of:

```text
Input Features
      ↓
Dense Layer
      ↓
Activation Function
      ↓
Dense Layer
      ↓
Activation Function
      ↓
Output Layer
      ↓
Sigmoid
      ↓
Churn Probability
```

The output represents the probability that a customer will churn.

For example:

```text
Prediction Probability = 0.82
```

can be interpreted as a high predicted probability of churn, subject to the application's chosen classification threshold.

## 🔧 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Streamlit**
* **Jupyter Notebook**
* **Pickle**

## 📁 Project Structure

```text
ANN-classification/
│
├── app.py
├── Churn_Modelling.csv
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

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/bharat-02/ANN-classification.git
```

### 2. Navigate to the project

```bash
cd ANN-classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

After installing the dependencies:

```bash
python -m streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

## 🖥️ Streamlit Application

The application allows users to enter customer information through an interactive interface.

Example inputs include:

```text
Geography          → Germany
Gender             → Male
Age                → 35
Balance            → 50000
Credit Score       → 650
Estimated Salary   → 75000
Tenure             → 5
Number of Products → 2
Has Credit Card    → 1
Is Active Member   → 1
```

The entered information is processed using the same preprocessing objects used during model development and passed to the trained ANN model.

## 🔄 Preprocessing

The project uses different preprocessing techniques for categorical and numerical features.

### One-Hot Encoding

`Geography` is transformed using a `OneHotEncoder`.

```python
onehot_encoder_geo
```

### Label Encoding

`Gender` is transformed using a `LabelEncoder`.

```python
label_encoder_gender
```

### Feature Scaling

Numerical features are scaled using the saved scaler:

```python
scaler.pkl
```

The preprocessing objects are saved and reused during prediction to ensure that the application applies the same transformations used during model training.

## 💾 Saved Model and Preprocessors

The repository contains the trained model and preprocessing objects:

```text
model.h5
scaler.pkl
onehot_encoder_geo.pkl
label_encoder_gender.pkl
```

These files allow the Streamlit application to make predictions without retraining the model every time the application starts.

## 📓 Notebooks

### `experiments.ipynb`

Contains the model development and experimentation workflow.

### `prediction.ipynb`

Contains the prediction-related workflow used for testing the trained model.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

**[ANN Customer Churn Prediction — Live Demo](https://ann-classification-msga88z4jmnzvharlhnwbf.streamlit.app/)**

## 🎯 Project Objectives

* Understand the customer churn prediction problem.
* Perform preprocessing of categorical and numerical data.
* Build an Artificial Neural Network for binary classification.
* Train and evaluate the ANN model.
* Save the trained model and preprocessing objects.
* Create an interactive Streamlit interface.
* Deploy the machine learning application online.

## 🔮 Future I
