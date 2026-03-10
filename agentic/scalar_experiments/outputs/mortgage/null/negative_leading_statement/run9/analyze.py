import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = "mortgage.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Drop index-like column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure expected columns
    required = [
        "female",
        "accept",
        "deny",
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Use accept as outcome; if accept missing, derive from deny (but accept exists here)
    df = df.copy()

    # Basic missingness handling: drop rows with NA in analysis columns
    analysis_cols = [
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
    df = df.dropna(subset=analysis_cols)

    # Unadjusted acceptance rates by gender
    grp = df.groupby("female")["accept"].agg(["mean", "count"]).rename(columns={"mean": "accept_rate"})
    # Difference female - male
    if 0 in grp.index and 1 in grp.index:
        rate_male = grp.loc[0, "accept_rate"]
        rate_female = grp.loc[1, "accept_rate"]
        n_male = int(grp.loc[0, "count"])
        n_female = int(grp.loc[1, "count"])
    else:
        raise ValueError("Both male (0) and female (1) groups required.")

    # Two-proportion z-test
    successes = np.array([
        df.loc[df["female"] == 0, "accept"].sum(),
        df.loc[df["female"] == 1, "accept"].sum(),
    ])
    ns = np.array([n_male, n_female])
    z_stat, p_val_prop = proportions_ztest(successes, ns)

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
    ]]
    X = sm.add_constant(X)
    y = df["accept"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Extract female coefficient
    coef = result.params["female"]
    se = result.bse["female"]
    p_val = result.pvalues["female"]
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    output = {
        "n": int(len(df)),
        "accept_rate_male": float(rate_male),
        "accept_rate_female": float(rate_female),
        "accept_rate_diff_female_minus_male": float(rate_female - rate_male),
        "prop_test_z": float(z_stat),
        "prop_test_p": float(p_val_prop),
        "logit_female_coef": float(coef),
        "logit_female_or": odds_ratio,
        "logit_female_or_ci95": [ci_low, ci_high],
        "logit_female_p": float(p_val),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
