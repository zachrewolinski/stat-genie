import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main():
    df = pd.read_csv("mortgage.csv")

    # Basic sanity checks
    required = ["female", "accept", "deny"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Use accept as outcome (1=accepted)
    y = df["accept"].astype(float)
    female = df["female"].astype(float)

    # Unadjusted difference in approval rates
    rate_female = y[female == 1].mean()
    rate_male = y[female == 0].mean()
    diff = rate_female - rate_male

    # Two-proportion z-test (approx) for difference in rates
    n_f = (female == 1).sum()
    n_m = (female == 0).sum()
    p_pool = (y[female == 1].sum() + y[female == 0].sum()) / (n_f + n_m)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_f + 1 / n_m))
    z = diff / se_pool if se_pool > 0 else np.nan
    p_unadj = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    # Adjusted logistic regression with key covariates
    # Select columns that are likely available and relevant to creditworthiness
    covariates = [
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
    covariates = [c for c in covariates if c in df.columns]
    X = df[covariates].astype(float)
    X = sm.add_constant(X, has_constant="add")

    # Fit logit; drop missing rows if any
    data = pd.concat([y, X], axis=1).dropna()
    y2 = data["accept"]
    X2 = data.drop(columns=["accept"])

    model = sm.Logit(y2, X2)
    result = model.fit(disp=False)

    # Extract female coefficient and p-value
    if "female" in result.params.index:
        coef_female = result.params["female"]
        p_female = result.pvalues["female"]
        odds_ratio = np.exp(coef_female)
    else:
        coef_female = np.nan
        p_female = np.nan
        odds_ratio = np.nan

    print("Unadjusted approval rate female:", rate_female)
    print("Unadjusted approval rate male:", rate_male)
    print("Unadjusted difference (female - male):", diff)
    print("Unadjusted p-value (two-proportion z):", p_unadj)
    print("Adjusted logit female coef:", coef_female)
    print("Adjusted logit female p-value:", p_female)
    print("Adjusted logit female odds ratio:", odds_ratio)


if __name__ == "__main__":
    main()
