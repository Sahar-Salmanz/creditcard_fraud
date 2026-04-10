# Credit Card Fraud Detection
An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, built on the <ins>Kaggle Credit Card Fraud Detection dataset</ins>. The notebook covers the full ML lifecycle, from raw data to explainable predictions, production monitoring, and a feedback loop for continuous improvement.


## Dataset
284,807 transactions made by European cardholders over two days in September 2013.
- Features `V1`–`V28` are PCA-transformed (original features are confidential)
- `Time` and `Amount` are the only non-PCA columns
- __Class imbalance:__ 492 frauds out of 284,807 transactions (~0.17%)

Download creditcard.csv from <ins>Kaggle</ins> and place it in the project directory before running the notebook.


## Notebook Structure
|Section|Description|
|-------|-----------|
|Environment Setup|Installs and imports|
|Data Ingestion & EDA|Schema validation, class balance, amount and time distributions|
|Feature Engineering|8 derived features from `Time` and `Amount`|
|Preprocessing|Chronological split, StandardScaler, SMOTE oversampling|
|Model Training|Logistic Regression, Random Forest, XGBoost|
|Evaluation|PR-AUC, ROC-AUC, confusion matrices, cost-sensitive analysis|
|Threshold Optimisation|F1-optimal threshold sweep on validation set|
|Explainability|SHAP global importance, beeswarm summary, local waterfall|
|Monitoring & Drift Detection|PSI per feature, score distribution shift|
|Feedback Loop|Active learning review queue, chargeback signal handling|


## Feature Engineering
The `V1`–`V28` PCA features are used as-is. The following features are derived from `Time` and `Amount`:

|Feature|Description|
|-------|-----------|
|`hour_of_day`|Fractional hour within a 24-hour cycle|
|`hour_sin`, `hour_cos`|Cyclic encoding to remove the midnight discontinuity|
|`is_night`|Binary flag for transactions between 22:00–06:00|
|`log_amount`|`log1p(Amount)` (corrects the heavy right skew)|
|`amount_bucket`|Quintile bin of `Amount` (0–4)|
|`amount_x_hour`|Interaction term: `log_amount × hour_of_day`|
|`tx_velocity_1h`|Rolling transaction count (proxy for velocity)|
|`rolling_avg_amount`|Rolling mean amount over the same window|


## Models
Three model tiers are trained and compared:
|Model|Role|
|-----|----|
|Logistic Regression|Interpretable baseline with calibrated probabilities|
|Random Forest|Robust ensemble with OOB score estimate|
|XGBoost|Primary model; `scale_pos_weight` handles class imbalance natively|


## Evaluation Approach
__Why PR-AUC instead of accuracy?__

With 0.17% fraud, a model that predicts everything as legitimate achieves 99.83% accuracy while catching zero fraud. Precision-Recall AUC directly measures minority-class performance and is the primary metric here.

__Cost-sensitive thresholding:__

Rather than using the default 0.5 cutoff, the decision threshold is swept over all values from the precision-recall curve and the one maximising F1 on the validation set is selected. This trades off:
- _False Negative (missed fraud)_: ~$120 average loss
- _False Positive (blocked legitimate transaction)_: ~$5 customer friction cost


## Explainability
__SHAP (SHapley Additive exPlanations)__ is used to explain both global and local model behaviour:
- __Global:__ mean absolute SHAP values identify the features driving fraud predictions across the dataset. V14, V17, and V12 are consistently the most discriminative PCA components
- __Local__: per-transaction waterfall plots show exactly which features pushed a specific prediction toward fraud or legitimate, supporting analyst review and regulatory compliance


## Monitoring
Population Stability Index (PSI) is computed per feature and for the score distribution to detect distribution shift between reference (validation) and production (test) data:
|PSI|Status|
|---|------|
|< 0.10|Stable|
|0.10 – 0.25|Watch|
|> 0.25|Retrain|



## Getting Started
1. __Clone the repo and install dependencies__
```
git clone https://github.com/Sahar-Salmanz/creditcard_fraud.git
cd creditcard_fraud
pip install -r requirements.txt
```

2. __Download the dataset__

Download `creditcard.csv` from <ins>Kaggle</ins> and place it in the project root.

3. __Run the notebook__
```
jupyter notebook credit_fraud.ipynb
```
Or open directly in VS Code, JupyterLab, or upload to Google Colab.



## Requirements
```
numpy
pandas
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
jupyter
```


## Project Structure
```
creditcard_fraud/
├── credit_fraud.ipynb   # Main notebook
├── requirements.txt
└── README.md
```