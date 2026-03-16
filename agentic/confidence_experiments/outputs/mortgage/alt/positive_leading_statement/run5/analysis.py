import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("mortgage.csv")

    # Basic cleanup
    df = df.copy()
    # Ensure binary ints
    for col in ["female", "accept", "deny", "black", "self_employed", "married", "bad_history", "denied_PMI"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Acceptance rate by gender
    grp = df.groupby("female")
    accept_rate = grp["accept"].mean()
    n_by_gender = grp.size()

    # Two-proportion z-test for acceptance rates
    counts = grp["accept"].sum().astype(int)
    nobs = n_by_gender.astype(int)
    # ensure order: female=0,1
    counts = counts.reindex([0.0, 1.0])
    nobs = nobs.reindex([0.0, 1.0])
    z_stat, p_val = proportions_ztest(counts, nobs)

    # Logistic regression with controls
    # Define covariates (exclude denied_PMI as it may be downstream of approval decision)
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
    ]
    # Keep only available columns
    covariates = [c for c in covariates if c in df.columns]

    model_df = df[["accept"] + covariates].dropna()
    y = model_df["accept"]
    X = model_df[covariates]
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False)
    except Exception:
        # fallback to regularized fit if needed
        result = logit_model.fit_regularized(disp=False)

    # Extract female coefficient info if available
    coef = result.params.get("female", np.nan)
    pvalue = result.pvalues.get("female", np.nan)
    # Odds ratio
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

    # Prepare outputs
    output = {
        "accept_rate_male": float(accept_rate.get(0.0, np.nan)),
        "accept_rate_female": float(accept_rate.get(1.0, np.nan)),
        "n_male": int(n_by_gender.get(0.0, 0)),
        "n_female": int(n_by_gender.get(1.0, 0)),
        "prop_test_z": float(z_stat),
        "prop_test_p": float(p_val),
        "logit_female_coef": float(coef) if np.isfinite(coef) else None,
        "logit_female_p": float(pvalue) if np.isfinite(pvalue) else None,
        "logit_female_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
