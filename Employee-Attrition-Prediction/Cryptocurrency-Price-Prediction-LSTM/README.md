# Employee Attrition Prediction System

A machine learning project that predicts whether an employee is likely to leave the company, built as part of the Algonive internship program.

## Problem Statement
Employee attrition (turnover) is costly for organizations. This project uses HR data to build a classification model that predicts which employees are at risk of leaving, helping HR teams take proactive steps.

## Dataset
- Source: IBM HR Analytics Employee Attrition dataset (public)
- ~1470 employee records
- Features: Age, Department, Monthly Income, Job Satisfaction, Overtime, Years at Company, and more
- Target: `Attrition` (Yes / No)

## Models Used
| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble model, best performance |

## Results
All three models are trained and evaluated. Metrics reported:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Charts Generated
- `chart_attrition_count.png` — Overall attrition distribution
- `chart_age_distribution.png` — Age distribution by attrition
- `chart_income_vs_attrition.png` — Monthly income vs attrition
- `chart_confusion_matrix.png` — Confusion matrix of best model
- `chart_feature_importance.png` — Top 10 important features

## How to Run

```bash
pip install -r requirements.txt
python train.py
```

## Project Structure
```
employee_attrition_project/
├── train.py               # Main script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── attrition_model.joblib # Saved best model (generated after run)
└── chart_*.png            # Charts (generated after run)
```

## Key Findings
- Overtime, Monthly Income, and Age are among the top predictors of attrition
- Employees who work overtime are significantly more likely to leave
- Random Forest consistently outperforms simpler models on this dataset

## Author
Internship Project — Algonive
