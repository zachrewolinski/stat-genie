import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Clean column names if needed
    df = df.copy()

    # Outcome: deny (1=denied, 0=accepted)
    y = df["deny"].astype(float)

    # Predictors
    predictors = [
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
    X = df[predictors].astype(float)
    X = sm.add_constant(X)

    # Drop rows with missing or infinite values
    data = pd.concat([y, X], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    y = data["deny"]
    X = data.drop(columns=["deny"])

    # Fit logistic regression
    model = sm.Logit(y, X)
    result = model.fit(disp=False, cov_type="HC1")

    # Raw denial rates by gender
    rates = df.groupby("female")["deny"].mean()

    # Extract female effect
    coef = result.params["female"]
    se = result.bse["female"]
    pval = result.pvalues["female"]
    odds_ratio = float(np.exp(coef))

    # Print key outputs
    print("Raw denial rates by gender (female=1, male=0):")
    print(rates)
    print("\nLogit (deny) with controls, robust SE (HC1):")
    print(f"female coef: {coef:.4f}, SE: {se:.4f}, p-value: {pval:.4g}, odds ratio: {odds_ratio:.4f}")

    # Also report confidence interval for odds ratio
    ci_low, ci_high = result.conf_int().loc["female"]
    or_low, or_high = np.exp(ci_low), np.exp(ci_high)
    print(f"female odds ratio 95% CI: [{or_low:.4f}, {or_high:.4f}]")

if __name__ == "__main__":
    main()
