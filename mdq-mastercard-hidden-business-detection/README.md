# Hidden Business Detection — MDQ × Mastercard Data Quest

**Task:** Identify "hidden entrepreneurs" — consumer cardholders running business activity through personal cards — using only transaction-level data and a small set of known business cards.

## Approach
- Framed as a **Positive-Unlabeled (PU) learning** problem: known business cards = labeled positives, all consumer cards = unlabeled pool (some of which are secretly businesses).
- Engineered 30+ behavioral features per card across 6 families: spending patterns, merchant/MCC concentration (HHI), B2B activity share, recurring large payments, cross-border activity, and business-hours activity.
- Trained a rank-averaged ensemble of LightGBM + XGBoost, explained with SHAP.
- Validated using a held-out business injection strategy (hiding known business cards among consumers to estimate real-task performance without access to true hidden labels).

##  Known limitation
Our validation approach returned a **perfect proxy AUC (1.0000) across all 5 folds** — a result too good to be trusted at face value. The competition's actual winning teams reported non-perfect scores on the true hidden labels, which indicates our validation setup likely didn't capture the real difficulty of the task: distinguishing genuinely disguised businesses from consumers who are simply high spenders.

Most likely cause: our synthetic validation (hiding known, "obviously business-like" cards among consumers) was an easier task than the real one, since genuinely hidden businesses probably resemble consumers much more closely than our held-out business cards did.

**This version (v1) is kept as-is intentionally**, to be transparent about the process. A corrected version (v2), which stress-tests validation with harder synthetic negatives and revisits feature leakage risk, is planned as a follow-up.

## Stack
Python, Pandas, NumPy, LightGBM, XGBoost, SHAP, Scikit-learn

## Result
Hackathon submission — ranked consumer cards by estimated likelihood of hidden business activity, generated for MDQ × Mastercard Data Quest.
