# Diabetes Risk Prediction

Binary classification project predicting diabetes risk from health indicator survey data, covering full EDA, preprocessing, multi-model training, and evaluation.

## Dataset
[CDC Diabetes Health Indicators (UCI ML Repository)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) — derived from the CDC's BRFSS 2015 survey, containing health and lifestyle indicators (BMI, blood pressure, cholesterol, physical activity, etc.) with a binary diabetes label.

## Approach
1. **EDA** — class imbalance check, feature distributions vs. target, correlation heatmap
2. **Preprocessing** — [briefly note what you actually did: scaling, encoding, handling imbalance, etc.]
3. **Model training** — Logistic Regression, Decision Tree / Random Forest, XGBoost/LightGBM, KNN, SVM
4. **Evaluation** — ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix, ROC & PR curves
5. **Insights** — comparison of model performance and discussion of results

## Key results
[Fill in: which model performed best, and on what metric — e.g. "XGBoost achieved the highest ROC-AUC (0.XX), outperforming Logistic Regression baseline (0.XX)"]

## Stack
Python, Pandas, Scikit-learn, XGBoost/LightGBM, Matplotlib, Seaborn
