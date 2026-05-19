# Telco Customer Churn Prediction

A machine learning model that predicts whether a telecom customer is likely to churn which is deployed as an interactive Streamlit web application.

**Test set performance:** 76% accuracy, **81% recall on the churn class**, F1 0.64 (churn).

---

## Problem

Telcos lose significant revenue to customer churn. Acquiring a new customer is far more expensive than retaining an existing one, so identifying at-risk customers early lets the business intervene with targeted retention offers.

This project takes the IBM Telco Customer Churn dataset (7,043 customers, 21 features) and trains a model that flags customers likely to leave, then exposes the model as a Streamlit app for non-technical stakeholders to use.

---

## Approach

### Data preprocessing

The raw dataset has mixed categorical and numeric features and a 27%/73% class imbalance (churn vs no-churn). Preprocessing steps:

- Coerce `TotalCharges` to numeric and drop rows with missing values (11 rows)
- Drop `customerID` (non-predictive)
- Collapse `"No internet service"` and `"No phone service"` into `"No"` across service columns
- Binary encode Yes/No columns and gender (Male = 1, Female = 0)
- One-hot encode `InternetService`, `Contract`, and `PaymentMethod`

Final feature matrix: **26 features**.

### Model selection

I compared two classifiers, each with and without SMOTEENN resampling for class imbalance:

| Model | Accuracy | Churn F1 | Churn Recall |
|---|---|---|---|
| Decision Tree (no resampling) | 0.78 | 0.57 | 0.51 |
| Random Forest (no resampling) | 0.79 | 0.57 | 0.50 |
| Decision Tree + SMOTEENN | 0.76 | 0.62 | 0.78 |
| **Random Forest + SMOTEENN** | **0.76** | **0.64** | **0.81** |

I also tried PCA-reduced features, which underperformed (72% accuracy), so it was dropped.

**Selected model: Random Forest + SMOTEENN.** It achieves the highest recall on the churn class (81%), which matters more than raw accuracy for a churn problem. The business cost of missing a churner is much higher than the cost of contacting a customer who would have stayed.

### Important methodology note

In my earlier iterations I applied SMOTEENN to the entire dataset *before* the train/test split. That inflated test scores to ~95% accuracy because the test set contained synthetic samples generated from the training distribution (data leakage). The numbers in this README are from the corrected pipeline: split first, then resample only the training set.

### Top predictive features

The Random Forest's most informative features (by impurity-based importance):

1. `Contract_Month-to-month` (26%) - month-to-month customers churn far more than contract customers
2. `tenure` (21%) - newer customers churn more
3. `InternetService_Fiber optic` (10%)
4. `PaymentMethod_Electronic check` (8%)
5. `TotalCharges` and `MonthlyCharges` (15% combined)

---

## Repository structure

- `streamlit_app.py` - interactive Streamlit prediction app
- `train.py` - reproducible training pipeline (preprocessing + SMOTEENN + Random Forest)
- `Telco_Customer_Churn_(eda).ipynb` - exploratory data analysis with visualisations
- `Telco_Customer_Churn_ML.ipynb` - initial exploratory pass comparing multiple classifiers (DT, RF, NB, KNN, SVM, LR)
- `Customer_Churn_ML_model.ipynb` - refined pass that selected the final model
- `churn_model.pkl` - trained Random Forest, produced by `train.py`
- `feature_columns.pkl` - feature column order the model expects
- `telco_churn.csv` - IBM Telco Customer Churn dataset
- `requirements.txt` - Python dependencies

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional - `churn_model.pkl` is already included)

```bash
python train.py
```

This prints the evaluation metrics and produces `churn_model.pkl` and `feature_columns.pkl`.

### 3. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`. Adjust customer details in the form and click **Predict** to get a churn probability.

---

## Tech stack

- **Modelling:** scikit-learn, imbalanced-learn (SMOTEENN), pandas, NumPy
- **App:** Streamlit
- **Persistence:** joblib

---

## What I'd do next

- **Hyperparameter tuning** with cross-validated grid search (current params are reasonable defaults but unoptimised)
- **Add an explanation layer** (SHAP values per prediction) so business users see *why* the model flagged a customer
- **Containerise with Docker** and deploy as a FastAPI service for programmatic batch predictions alongside the Streamlit UI
- **Track model performance over time** - churn drivers shift, so the model would need retraining on fresh data quarterly

---

## Author

**Tharakesh Aravindan S.T** — [LinkedIn](https://www.linkedin.com/in/tharakesharavindan-96054422b) — [GitHub](https://github.com/Tharakesh-Aravindan)

Original project completed during a 2022 internship at DeepSphere.AI; refactored and corrected in 2026 with proper train/test methodology and a working deployment.
