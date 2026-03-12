import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning: drop rows with missing key variables
    cols_needed = [
        "female",
        "accept",
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
    df = df.dropna(subset=cols_needed)

    # Ensure binary columns are 0/1 numeric
    for col in ["female", "accept", "black", "self_employed", "married", "bad_history", "denied_PMI"]:
        df[col] = df[col].astype(int)

    # Approval rates by gender
    rate_by_gender = df.groupby("female")["accept"].mean()
    n_by_gender = df.groupby("female")["accept"].size()

    # Contingency table for chi-square test
    # Rows: female (0 male, 1 female); Cols: accept (0 denied, 1 accepted)
    contingency = pd.crosstab(df["female"], df["accept"])
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)

    # Logistic regression with controls
    X = df[[
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
    ]].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["accept"]

    logit = sm.Logit(y, X)
    result = logit.fit(disp=False)

    coef_female = result.params["female"]
    se_female = result.bse["female"]
    p_female = result.pvalues["female"]
    odds_ratio = np.exp(coef_female)

    # Compute average marginal effect for female (discrete change 0->1)
    # Using statsmodels' get_margeff with method 'dydx'
    marg = result.get_margeff(at="overall", method="dydx")
    marg_female = float(marg.margeff[marg.summary_frame().index.get_loc("female")])
    marg_p = float(marg.pvalues[marg.summary_frame().index.get_loc("female")])

    print("Rows used:", len(df))
    print("Approval rate by gender (female=0 male, 1 female):")
    print(rate_by_gender)
    print("Counts by gender:")
    print(n_by_gender)
    print("Contingency table (female x accept):")
    print(contingency)
    print(f"Chi-square test p-value: {p_chi2:.6g}")
    print("\nLogit with controls: accept ~ female + covariates")
    print(f"female coef: {coef_female:.6g}, SE: {se_female:.6g}, p: {p_female:.6g}")
    print(f"female odds ratio: {odds_ratio:.6g}")
    print(f"female avg marginal effect (dP): {marg_female:.6g}, p: {marg_p:.6g}")


if __name__ == "__main__":
    main()
