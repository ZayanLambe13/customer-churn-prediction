# Customer Churn Prediction using Machine Learning

## 📌 Problem Statement
Customer churn refers to customers discontinuing a company’s service. Retaining existing customers is significantly more cost-effective than acquiring new ones.  
The objective of this project is to predict customer churn using machine learning and identify key factors that drive churn, enabling proactive retention strategies.

---

## 📊 Dataset
- **Source:** IBM Telco Customer Churn Dataset (UCI Machine Learning Repository)
- **Size:** ~7,000 customers
- **Target Variable:** `Churn` (Yes / No)
- **Features:** Customer demographics, account information, services used, and billing details

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights obtained from EDA:
- Customers on **month-to-month contracts** churn significantly more than long-term contract customers
- **Short tenure** customers are more likely to churn
- Customers with **higher monthly charges** show higher churn probability
- **Fiber optic internet users** have higher churn rates compared to DSL users

---

## 🧪 Feature Engineering
- Converted `TotalCharges` to numeric and handled missing values
- Encoded target variable (`Churn`) into binary format
- Created new features:
  - `tenure_group` (customer tenure buckets)
  - `avg_monthly_spend` (behavioral spending feature)
- Applied one-hot encoding for categorical variables
- Removed non-informative columns (Customer ID)

---

## 🤖 Modeling & Evaluation
### Models Used
- Logistic Regression (baseline, interpretable)
- Random Forest Classifier (non-linear model)

### Evaluation Metrics
- **Recall** (prioritized to avoid missing churners)
- **ROC-AUC**
- Confusion Matrix

### Results
- Logistic Regression provided a strong interpretable baseline
- Random Forest improved predictive performance by capturing feature interactions
- Contract type, tenure, and monthly charges emerged as key churn drivers

---

## 📈 Key Business Insights
- Long-term contracts significantly reduce churn risk
- Early-stage customers need targeted engagement
- Pricing sensitivity plays a major role in churn behavior

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📂 Project Structure
