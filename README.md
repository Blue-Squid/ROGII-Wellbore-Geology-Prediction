# ROGII Wellbore Geology Prediction: A Machine Learning Approach to Subsurface Mapping

## Abstract

Briefly summarize the objective (predicting `tvt` along a horizontal wellbore), the methodologies applied, and the final cross-validation vs. public leaderboard RMSE score.

## 1. Introduction

Detail the industrial problem. Reference the high cost of redundant drilling and the limitations of raw seismic/logging tools. Define the input variables and the target (`tvt`).

## 2. Methodology

### 2.1 Cross-Validation Strategy

Explain the grouping strategy used to prevent spatial data leakage.

### 2.2 Feature Engineering

List the mathematical and domain-specific transformations applied to the raw data (e.g., rolling means, gradient calculations).

### 2.3 Model Architecture

Describe the models utilized (e.g., "An ensemble approach utilizing Gradient Boosted Trees and a 1D-CNN spatial sequence model...").

## 3. Experimental Setup

- **Hardware:** Local NVIDIA RTX 5080 laptop variant.
- **Frameworks:** Polars, PyTorch, XGBoost.
- **Hyperparameter Optimization:** Optuna (50 trials, TPE Sampler).

## 4. Results & Analysis

| Model              | Local CV RMSE | Kaggle Public LB RMSE | Inference Time |
| :----------------- | :------------ | :-------------------- | :------------- |
| Baseline (LGBM)    | 0.00          | 0.00                  | 2 mins         |
| Tuned XGBoost      | 0.00          | 0.00                  | 5 mins         |
| 1D-CNN Sequence    | 0.00          | 0.00                  | 12 mins        |
| **Final Ensemble** | **0.00**      | **0.00**              | **15 mins**    |

## 5. Conclusion

Summarize findings, feature importances, and potential real-world applications for automated drilling systems.
