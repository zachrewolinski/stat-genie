import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = "mortgage.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleanup
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing key fields
    key_cols = ["female", "accept", "deny"]
    df = df.dropna(subset=key_cols)

    n_total = len(df)
    n_female = int((df["female"] == 1).sum())
    n_male = int((df["female"] == 0).sum())

    # Check accept/deny consistency
    consistency = None
    if "accept" in df.columns and "deny" in df.columns:
        consistency = float((df["accept"] == (1 - df["deny"]).astype(int)).mean())

    # Raw acceptance rates
    accept_female = df.loc[df["female"] == 1, "accept"].mean()
    accept_male = df.loc[df["female"] == 0, "accept"].mean()
    diff = accept_female - accept_male

    # Two-proportion z-test
    count = np.array([
        df.loc[df["female"] == 1, "accept"].sum(),
        df.loc[df["female"] == 0, "accept"].sum(),
    ])
    nobs = np.array([n_female, n_male])
    z_stat, z_p = proportions_ztest(count, nobs)

    # Logistic regression (adjusted)
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

    model_df = df.dropna(subset=["accept"] + covariates)

    X = model_df[covariates]
    X = sm.add_constant(X, has_constant="add")
    y = model_df["accept"]

    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False)
    except Exception:
        res = logit.fit(disp=False, method="newton", maxiter=200)

    coef_female = res.params.get("female", np.nan)
    p_female = res.pvalues.get("female", np.nan)
    conf_int = res.conf_int().loc["female"].tolist() if "female" in res.params.index else [np.nan, np.nan]

    or_female = float(np.exp(coef_female)) if np.isfinite(coef_female) else np.nan
    or_ci = [float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))] if np.all(np.isfinite(conf_int)) else [np.nan, np.nan]

    results = {
        "n_total": n_total,
        "n_female": n_female,
        "n_male": n_male,
        "accept_rate_female": float(accept_female),
        "accept_rate_male": float(accept_male),
        "accept_rate_diff": float(diff),
        "z_stat": float(z_stat),
        "z_p": float(z_p),
        "accept_deny_consistency": consistency,
        "logit_coef_female": float(coef_female),
        "logit_p_female": float(p_female),
        "logit_or_female": float(or_female),
        "logit_or_ci_low": float(or_ci[0]),
        "logit_or_ci_high": float(or_ci[1]),
        "logit_n": int(model_df.shape[0]),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
