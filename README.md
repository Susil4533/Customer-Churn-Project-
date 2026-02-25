# Customer Churn Prediction

A hypothesis-driven machine learning project to identify customers at high risk of churning and translate that risk into actionable retention strategies.

---

## Why This Project Exists

Acquiring a new customer costs significantly more than retaining an existing one. In competitive markets, churn is one of the most direct threats to business sustainability. The goal here is not just to build a model that predicts churn, but also is to build something that actually helps a business decide *who to act on, and when*.

This distinction matters. A model with 90% accuracy that misses most churners is useless in practice. This project is built around that reality.

---

## The Approach

Most churn projects jump straight to modeling. This one does not.

Before writing a single line of model code, two business hypotheses were formed based on domain reasoning:

**Hypothesis 1 — Contract Type**
Customers on month-to-month contracts churn more than those on long-term contracts because they face lower switching costs and have made less of a commitment.

**Hypothesis 2 — Payment Method**
Customers using non-automated payment methods churn more than those on automated payments because automated billing creates behavioral lock-in.

These hypotheses were tested statistically first, then the findings were carried directly into feature engineering and model interpretation. This creates a traceable chain from business question → data evidence → model decision.

---

## What the Data Shows

The dataset contains 7,032 telecom customers with 20 features covering demographics, contract details, services used, and billing information. About 26.6% of customers churned.

**Hypothesis 1 result — Contract Type vs Churn:**

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Chi-square statistic: 1179.5 — p-value: effectively zero. The relationship is statistically significant and practically dramatic.

**Hypothesis 2 result — Payment Method vs Churn:**

| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.2% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.3% |

Chi-square statistic: 645.4 — p-value: effectively zero. Notably, it is not simply automated vs non-automated. Electronic check customers churn at more than double the rate of mailed check users — suggesting a distinct behavioral profile worth targeting separately.

---

## Project Structure

```
Customer-Churn-Project/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   ├── processed_churn.csv                     # after cleaning
│   └── featured_churn.csv                      # model-ready features
│
├── notebooks/
│   ├── 01_eda.ipynb                            # EDA + hypothesis testing
│   ├── 02_feature_engineering.ipynb            # feature transformation
│   └── Modeling/
│       ├── 03a_logistic_regression.ipynb       # primary model
│       ├── 03b_random_forest.ipynb             # benchmark
│       ├── 03c_gradient_boosting.ipynb         # benchmark
│       └── 04_evaluation.ipynb                 # comparison + business insights
│
├── src/
│   ├── data_preprocessing.py                   # cleaning pipeline
│   ├── feature_engineering.py                  # transformation pipeline
│   ├── modeling.py                             # shared training utilities
│   └── evaluation.py                           # comparison + risk scoring
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── requirements.txt
└── README.md
```

---

## Feature Engineering Decisions

Every transformation was made deliberately, not by default.

**Contract type** was ordinal encoded (0, 1, 2) rather than one-hot encoded. This is because the hypothesis testing confirmed a natural commitment gradient — encoding it ordinally preserves that relationship in the model.

**Payment method** was one-hot encoded, but a derived feature `PaymentMethod_AutoPay` was also added as a binary flag. This directly operationalizes the behavioral lock-in concept from Hypothesis 2 and gives the model a cleaner signal to work with.

**Binary columns** (Yes/No fields) were label encoded. **Multi-class columns** were one-hot encoded with `drop_first=True` to prevent multicollinearity in logistic regression.

**Numerical features** (tenure, MonthlyCharges, TotalCharges) were standardized using StandardScaler, since logistic regression is sensitive to feature magnitude.

**Class imbalance** (26.6% churn) was handled using `class_weight='balanced'`, which adjusts the model to penalize missed churners proportionally. This was preferred over oversampling for simplicity and reproducibility.

---

## Modeling

Three models were trained and compared. Logistic Regression was chosen as the primary model — not because it is the most powerful, but because this project prioritizes decision support over raw performance.

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Gradient Boosting | 0.8404 | 0.7968 | 0.5017 | 0.6157 | 0.7356 |
| **Logistic Regression** | **0.8353** | **0.7941** | **0.4901** | **0.6061** | **0.7257** |
| Random Forest | 0.8134 | 0.4813 | 0.6207 | 0.5422 | 0.7839 |

**Why Logistic Regression over Gradient Boosting?**

The AUC difference between the two is 0.005 and the Recall difference is 0.003 — both negligible. Given that gap, the interpretability of Logistic Regression (coefficients that directly map to feature influence) is worth more than a marginal performance gain. A model you can explain to a business stakeholder is more useful than one you cannot.

**Why not Random Forest as primary?**

Random Forest achieved the highest accuracy (78.4%) but the lowest recall (0.48) — meaning it missed roughly half of all actual churners. In a churn problem, a missed churner is a lost customer. Accuracy without recall is misleading.

---

## From Model to Business Action

The model assigns a churn probability to each customer. Customers are then grouped into three risk tiers:

| Risk Tier | Probability Threshold | Recommended Action |
|---|---|---|
| High | > 0.70 | Immediate personal outreach + retention offer |
| Medium | 0.40 – 0.70 | Soft engagement — loyalty rewards |
| Low | < 0.40 | No action — routine monitoring |

**Top 20% targeting:**

By focusing retention efforts on the top 20% highest-risk customers, the business reaches a group where the actual churn rate is **66.2%** — compared to 26.6% overall. This is a 2.5x concentration, meaning the same retention budget covers significantly more at-risk customers.

---

## Limitations

No project is complete without acknowledging what it cannot do.

The associations found here are correlational, not causal. A customer using electronic check may churn for reasons entirely unrelated to their payment method — the feature is a proxy, not a cause.

The dataset does not include pricing history, service quality scores, or customer support interactions. These are likely strong drivers of churn that this model cannot account for.

The model is also trained on a static snapshot. Customer behavior changes over time, and a model that does not account for behavioral drift will degrade in real-world use.

---

## Future Work

- **Survival analysis** — model time-to-churn rather than binary churn outcome
- **Cost-sensitive learning** — incorporate actual retention cost vs. customer lifetime value to optimize intervention threshold
- **Time-series modeling** — capture how customer behavior evolves over their tenure
- **Threshold optimization** — tune the decision threshold based on a cost matrix rather than defaulting to 0.5

---

## How to Run

```bash
# Clone the repo
https://github.com/Susil4533/Customer-Churn-Project-.git
cd Customer-Churn-Project-

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
# 01_eda.ipynb → 02_feature_engineering.ipynb → 03a/b/c → 04_evaluation.ipynb
```

---

## Tech Stack

- **Python 3.12**
- **pandas, NumPy** — data manipulation
- **scikit-learn** — modeling and evaluation
- **Matplotlib, Seaborn** — visualization
- **Jupyter Notebook** — analysis environment
- **joblib** — model persistence

---

## Dataset

IBM Telco Customer Churn dataset — publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
7,032 customers, 20 features, binary churn target.
