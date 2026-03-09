# 📞 Telco Customer Churn Prediction --- End-to-End Machine Learning Project

An end-to-end machine learning project that analyzes telecom customer
behavior and predicts **customer churn** using multiple classification
models.

The goal of this project is to help telecom companies **identify
customers likely to leave** so they can take **preventive retention
actions**.

------------------------------------------------------------------------

# 📌 Business Problem

Customer churn is one of the biggest challenges in the
telecommunications industry.

Acquiring a new customer costs **5--10x more** than retaining an
existing one.

This project analyzes customer behavior and builds a predictive system
to identify **customers at risk of leaving the service**.

The model uses the **IBM Telco Customer Churn dataset containing 7,043
customers**.

The final solution enables:

-   📊 Data-driven retention strategies
-   🎯 Targeted marketing campaigns
-   💰 Reduced revenue loss from churn

------------------------------------------------------------------------

# 📊 Exploratory Data Analysis (EDA)

Key insights discovered during analysis:

### Contract Type

Customers with **Month-to-Month contracts** show the highest churn
probability.

### Payment Method

Customers using **Electronic Check** have significantly higher churn
compared to automatic payment methods.

### Tenure

Customers with **1--6 months tenure** are most likely to churn.

### Internet Service

Customers using **Fiber Optic** services show higher churn, indicating
possible pricing or service dissatisfaction.

------------------------------------------------------------------------

# ⚙️ Machine Learning Pipeline

## 1️⃣ Data Cleaning & Preprocessing

Key preprocessing steps:

**Feature Removal** Removed high-cardinality and irrelevant columns:

-   CustomerID
-   Latitude
-   Longitude

**Handling Missing Values**

-   TotalCharges converted to numeric
-   Null values handled appropriately

**Feature Encoding**

Categorical variables encoded using one-hot encoding.

**Feature Scaling**

Numerical features scaled using **StandardScaler**.

Important scaled features:

-   Tenure Months
-   Monthly Charges
-   Total Charges

Scaling improves performance for models like **SVM and KNN**.

------------------------------------------------------------------------

# 🤖 Models Implemented

The following machine learning models were trained and evaluated:

  Model                             Purpose
  --------------------------------- -----------------------------------
  Logistic Regression               Interpretable baseline model
  Random Forest                     Ensemble model with high accuracy
  Support Vector Classifier (SVC)   Handles complex boundaries
  Decision Tree                     Tree-based interpretable model
  K-Nearest Neighbors (KNN)         Distance-based classifier
  Gaussian Naive Bayes              Probabilistic classifier

Final selected model:

**Logistic Regression**

------------------------------------------------------------------------

# 📈 Model Performance

The model was evaluated using:

-   Accuracy
-   Confusion Matrix
-   Classification Report
-   ROC-AUC Score

These metrics ensure the model performs well on **unseen data**.

------------------------------------------------------------------------

# 🚀 Deployment Ready

The trained model is serialized using **Joblib** so it can be integrated
into production applications such as **Streamlit dashboards or APIs**.

### Saved Files

    churn_model.pkl
    model_columns.pkl

### Example Usage

``` python
import joblib

model = joblib.load("churn_model.pkl")
features = joblib.load("model_columns.pkl")

prediction = model.predict(preprocessed_input[features])
```

------------------------------------------------------------------------

# 🖥️ Streamlit Web Application

A **Streamlit interface** allows users to input customer data and
predict churn instantly.

Features:

-   Interactive input form
-   Real-time churn prediction
-   Clean UI
-   Business-friendly results

Run locally:

    streamlit run app.py

------------------------------------------------------------------------

# 📂 Project Structure

    Telco-Churn-Prediction
    │
    ├── data
    │   └── Telco_customer_churn.xlsx
    │
    ├── models
    │   ├── churn_model.pkl
    │   └── model_columns.pkl
    │
    ├── main.ipynb
    │
    ├── app.py
    │
    └── README.md

------------------------------------------------------------------------

# 🛠 Tech Stack

### Programming

-   Python

### Data Analysis

-   Pandas
-   NumPy

### Visualization

-   Matplotlib
-   Seaborn

### Machine Learning

-   Scikit-Learn

### Deployment

-   Streamlit
-   Joblib

------------------------------------------------------------------------

# 🎯 Key Skills Demonstrated

-   Exploratory Data Analysis (EDA)
-   Data Cleaning & Feature Engineering
-   Machine Learning Model Comparison
-   Model Evaluation
-   Model Serialization
-   Building Interactive ML Applications
-   End-to-End ML Workflow

------------------------------------------------------------------------

# 👨‍💻 Author

**Bilal Mahmood Ansari**

BS Computer Science Student\
Aspiring Data Analyst / Data Scientist

GitHub: https://github.com/Ansari2004

------------------------------------------------------------------------

# ⭐ If you found this project helpful

Please **star the repository** ⭐
