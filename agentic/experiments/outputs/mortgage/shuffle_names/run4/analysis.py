import pandas as pd
import statsmodels.api as sm
import numpy as np

DATA_PATH = "mortgage.csv"

# Based on metadata:
# - denied_PMI: 1 if applicant is female, 0 if male
# - deny: 1 if application was accepted, 0 if denied (approval indicator)
GENDER_COL = "denied_PMI"
APPROVAL_COL = "deny"

# Controls chosen to reflect core affordability/credit ratios without leaking the outcome
CONTROL_COLS = [
    "mortgage_credit",          # housing expense ratio proxy (per metadata)
    "housing_expense_ratio",    # debt-to-income style ratio
    "Unnamed: 0",               # loan-to-value ratio (per metadata)
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Basic approval rates by gender
    rates = df.groupby(GENDER_COL)[APPROVAL_COL].mean()
    rate_female = rates.get(1, np.nan)
    rate_male = rates.get(0, np.nan)
    rate_diff = rate_female - rate_male

    # Logistic regression: approval on gender + controls
    cols = [APPROVAL_COL, GENDER_COL] + CONTROL_COLS
    data = df[cols].dropna()
    X = sm.add_constant(data[[GENDER_COL] + CONTROL_COLS], has_constant="add")
    y = data[APPROVAL_COL]
    model = sm.Logit(y, X)
    res = model.fit(disp=False)

    coef = res.params[GENDER_COL]
    pval = res.pvalues[GENDER_COL]
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = np.exp(res.conf_int().loc[GENDER_COL])

    print("Approval rate (female=1):", rate_female)
    print("Approval rate (male=0):", rate_male)
    print("Difference (female - male):", rate_diff)
    print("Logit coef (female):", coef)
    print("Odds ratio (female):", odds_ratio)
    print("95% CI (odds ratio):", (float(ci_low), float(ci_high)))
    print("p-value (female):", pval)


if __name__ == "__main__":
    main()
