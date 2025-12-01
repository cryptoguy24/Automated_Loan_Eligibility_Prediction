# 🏦 Loan Eligibility Prediction: Automated Credit Risk Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Deployment--Ready-success)]()

> *"Transforming manual underwriting into a data-driven, real-time decision engine."*

---

## 📖 Executive Summary
In the banking sector, **Non-Performing Assets (NPAs)**—loans that default—are the primary threat to profitability. Traditional manual underwriting is slow, prone to human bias, and inefficient at scale.

This project implements an end-to-end **Machine Learning Pipeline** to predict loan eligibility (`Approved` vs. `Rejected`). By automating the initial risk assessment, we aim to:
1.  **Minimize Default Risk:** By accurately identifying high-risk applicants.
2.  **Accelerate Turnaround Time:** Reducing decision time from days to milliseconds.
3.  **Ensure Fairness:** Removing human bias from the approval process.

---

## ⚙️ Solution Architecture

The project is structured into three distinct phases, mimicking a real-world Data Science lifecycle:

| Phase | Notebook | Key Activities |
| :--- | :--- | :--- |
| **1. Analysis** | `notebook/1. EDA.ipynb` | Univariate/Bivariate analysis, identifying the "Co-applicant Paradox", and defining the customer persona. |
| **2. Development** | `notebook/2. Data_preprocessing_ML_Evaluation.ipynb.ipynb` | Pipeline construction (Imputation, Encoding), screening **14 algorithms**, and Hyperparameter Tuning via `GridSearchCV`. |
| **3. Evaluation** | `notebook/3. Model_Training_Test.ipynb` | Retraining the champion model (SVM) on consolidated data, final testing on unseen data, and serialization (`.pkl`). |

---

## 📊 Key Business Insights (EDA)
Before writing a single line of modeling code, we discovered critical patterns in the data:

### 1. The "Gatekeeper" vs. The "Sizer"
Our analysis revealed a two-stage decision hierarchy:
* **Credit History** is the **Gatekeeper**. If this is poor (`0.0`), rejection is almost guaranteed (99%), regardless of income.
* **Income & Education** are the **Sizers**. They do not determine *if* you get a loan, but *how much* (Loan Amount) you can afford.

### 2. The Co-Applicant Paradox
Initially, having a co-applicant seemed to increase rejection rates. Feature engineering revealed that co-applicants are often added to support weak primary applications. By creating a `Total_Income` feature, we resolved this multicollinearity and improved predictive power.

---

## 🏆 Model Performance
After screening 14 models (including Random Forest, Gradient Boosting, and XGBoost), the **Support Vector Machine (SVM)** emerged as the champion.

| Metric | Score | Why it matters? |
| :--- | :--- | :--- |
| **Accuracy** | **~85%** | Correctly classifies the majority of applications. |
| **Recall (Class 1)**| **99%** | **Crucial:** We almost never reject a valid, creditworthy customer (Opportunity Capture). |
| **False Negatives** | **1** | Only 1 eligible customer was incorrectly rejected in the test set. |

**Champion Hyperparameters:**
* **Algorithm:** `SVC (Support Vector Classifier)`
* **Kernel:** `rbf` (Radial Basis Function)
* **Regularization (C):** `1`
* **Gamma:** `auto`

---

## 🛠️ Project Structure
```text
├── data/
│   ├── Loan.csv                 # Raw dataset
│   ├── preprocessed_loan.csv    # Cleaned data (Phase 1 Output)
│   ├── X_train.csv, X_test.csv, X_valid.csv  # Split Features
│   └── y_train.csv, y_test.csv, y_valid.csv  # Split Targets
│
├── model/
│   └── Best_model.pkl           # Serialized Champion Model (SVM Pipeline)
│
├── notebook/
│   ├── 1. EDA.ipynb
│   ├── 2. Data_preprocessing_ML_Evaluation.ipynb.ipynb
│   └── 3. Model_Training_Test.ipynb
│
├── plots/                       # Generated Analysis Charts
│   ├── 1.univariate_numeric_cols.png
│   ├── 8.Bivariate_categorical_vs_target.png
│   └── ... (other EDA visualizations)
│
├── readme.md                    # Project Documentation
└── tree_structure.txt           # Directory Tree
└── demo.py                      # Demo model 