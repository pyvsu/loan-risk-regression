# Loan Risk Score Prediction

Predicts a borrower's risk score (higher = riskier) from loan application data, so a lending company's credit risk team can screen and price applicants quickly and consistently.

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

Most of the work was deciding what to leave out.

| Removed | Why |
|:---|:---|
| `LoanApproved` | Leakage. The decision is made from the risk score (approved loans average 40.1 risk vs 54.1 for rejected) |
| `InterestRate`, `BaseInterestRate`, `MonthlyLoanPayment` | Leakage. Set from, or built from, the risk judgement |
| `TotalDebtToIncomeRatio` | Leakage. It includes the loan payment (gap test 0.86) |
| `NetWorth`, `MonthlyIncome` | They repeat other columns (assets minus liabilities, and income divided by 12) |
| `ApplicationDate` | A date, not a driver of risk |
| `Experience` | Near-duplicate of `Age` (r = 0.98, VIF about 30). Kept `Age` because it is easier to explain |
| 12 weak features | No statistically clear effect (p above 0.05). Removing them did not change the test score |

- **Leakage test:** with the leaky columns, cross-validated R² was 0.813. Without them it was 0.642, the honest baseline for numeric columns before encoding.
- **Log transforms** on the six money columns were tested and skipped. They raised R² by only 0.004, below the 0.005 bar, and made coefficients harder to explain.


## Results

The final model (v3, 25 features) cuts the error roughly in half compared with always guessing the average.

| Model | Features | R² | RMSE | MAE |
|:---|--:|--:|--:|--:|
| Baseline (predict the average) | 0 | -0.002 | 7.889 | 6.208 |
| v1 (all cleaned features) | 38 | 0.759 | 3.871 | 2.964 |
| **v3 (final)** | **25** | **0.759** | **3.871** | **2.964** |

All scores are on a held-out test set (4,000 of 20,000 rows). The average miss is about 3 points, roughly 5% of the score range (28.8 to 84), which meets the success target of under about 10%. v3 scores the same as v1 with 13 fewer features, so it is simpler and just as accurate.


## Key findings

Each effect holds the other features equal.

- **Bankruptcy history** is the biggest single driver: about **+13.2 points**.
- **A past loan default** adds about **+6.9 points**.
- **Debt-to-income ratio:** each 0.1 rise adds about **+1.5 points**.
- **Annual income:** each extra 10,000 in income lowers risk by about **0.88 points**.
- **Total assets:** each extra 10,000 in assets lowers risk by about **0.20 points**.

These are patterns in this dataset. They do not prove cause and effect.
## Demo

![App screenshot](docs/app.png)

The app also shows a Low / Medium / High risk band based on training-data terciles.

## Limitations

- **Synthetic data.** Real borrowers will behave differently, so test on real data before relying on the model.
- **Association, not causation.** A coefficient shows a pattern in this data, not a cause.
- **Weak at the low end.** At the low end of the score range, a straight line can overshoot. 3 of 4,000 test rows missed by more than 25 points, and every other miss was 15.3 or less.
- **The score may come from a formula.** That would make it easy for a line to learn, so the fit may look better than it would on real data.

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
