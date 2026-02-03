import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    df = pd.read_csv("mortgage.csv")

    # Basic approval rate comparison by gender (unadjusted)
    approval_rates = df.groupby("female")["accept"].mean()
    contingency = pd.crosstab(df["female"], df["accept"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression of acceptance on gender + credit controls
    cols = [
        "accept",
        "female",
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    df_model = df[cols].dropna()
    X = sm.add_constant(df_model.drop(columns=["accept"]))
    y = df_model["accept"]
    model = sm.Logit(y, X).fit(disp=False)

    female_coef = model.params["female"]
    female_p = model.pvalues["female"]
    odds_ratio = float(np.exp(female_coef))

    # Print a compact summary for reproducibility
    print("Unadjusted approval rates (female=0,1):")
    print(approval_rates)
    print("Chi-square p-value (gender vs acceptance):", chi2_p)
    print("Logit female coef:", female_coef)
    print("Logit female p-value:", female_p)
    print("Female odds ratio:", odds_ratio)


if __name__ == "__main__":
    main()
