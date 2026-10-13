# Loan Risk Score Prediction

[One sentence: what it predicts and for whom.] **[Live app](PASTE_LINK)**

## Problem

A credit risk team at a lending company needs to screen and price loan applicants quickly and consistently. This project predicts a borrower's RiskScore from loan application data using linear regression. Success means beating the "predict the average" baseline and keeping the average error under about 10% of the score range. The data is synthetic, the results show relationships and not causes, and the score may come from a formula, so the model should be tested on real data before anyone relies on it.

## Data

- Source: [Financial Risk for Loan Approval (Kaggle)](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)
- 20,000 rows, 36 columns
- **Synthetic data.** Results show method, not real-world performance.
- The CSV is not in this repo. Download it and place it in `data/Loan.csv`.

## Approach

1. Cleaned the data and encoded categories
2. Removed leakage and duplicate columns
3. Explored the data (EDA)
4. Built and compared linear regression models
5. Deployed a Streamlit app

## Key decisions

| Dropped | Why |
|:---|:---|
| [column] | [reason] |

## Results

| Model | R² | RMSE | MAE |
|:---|--:|--:|--:|
| Baseline | [ ] | [ ] | [ ] |
| Final (v3, 25 features) | [ ] | [ ] | [ ] |

## Key findings

- [finding 1]
- [finding 2]
- [finding 3]

## Demo

![App screenshot](docs/app.png)

The app also shows a Low / Medium / High risk band based on training-data terciles.

## Limitations

- [limit 1]
- [limit 2]
- [limit 3]

## How to run

```bash
git clone https://github.com/pyvsu/loan-risk-regression.git
cd loan-risk-regression
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# download Loan.csv from Kaggle into data/
streamlit run app.py
```

## Repo structure

```
loan-risk-regression/
├─ data/          (not uploaded)
├─ notebooks/     (01 to 04, step by step)
├─ models/        (saved model + metadata)
├─ predict.py
├─ app.py
├─ requirements.txt
└─ README.md
```

## Next steps

- [idea 1]
- [idea 2]
