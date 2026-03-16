import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


def main():
    df = pd.read_csv("mortgage.csv")

    # Basic cleanup
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure binary columns are numeric ints
    for col in ["female", "accept", "deny"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Contingency table for approval by gender
    ct = pd.crosstab(df["female"], df["accept"])
    # Expect columns 0,1
    female_accept_rate = df.loc[df["female"] == 1, "accept"].mean()
    male_accept_rate = df.loc[df["female"] == 0, "accept"].mean()

    # Chi-square test of independence
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

    # Two-proportion z-test (female vs male acceptance)
    # counts of accept=1
    fem_accept = df.loc[df["female"] == 1, "accept"].sum()
    fem_n = (df["female"] == 1).sum()
    male_accept = df.loc[df["female"] == 0, "accept"].sum()
    male_n = (df["female"] == 0).sum()
    count = np.array([fem_accept, male_accept])
    nobs = np.array([fem_n, male_n])
    stat_z, p_z = sm.stats.proportions_ztest(count, nobs)

    # Missingness overview
    missing = df.isna().sum().to_dict()

    # Logistic regression: accept ~ female (unadjusted)
    logit_unadj = None
    try:
        data_unadj = df[["accept", "female"]].dropna()
        X_unadj = sm.add_constant(data_unadj[["female"]])
        y = data_unadj["accept"]
        logit_unadj = sm.Logit(y, X_unadj).fit(disp=0)
    except Exception as exc:
        logit_unadj = exc

    # Logistic regression: accept ~ female + controls (adjusted)
    controls = [
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
    available_controls = [c for c in controls if c in df.columns]
    logit_adj = None
    try:
        X_adj = sm.add_constant(df[["female"] + available_controls])
        data_adj = pd.concat([df[["accept"]], X_adj], axis=1).dropna()
        y_adj = data_adj["accept"]
        X_adj = data_adj.drop(columns=["accept"])
        logit_adj = sm.Logit(y_adj, X_adj).fit(disp=0)
    except Exception as exc:
        logit_adj = exc

    summary = {
        "n_total": int(len(df)),
        "female_n": int(fem_n),
        "male_n": int(male_n),
        "female_accept_rate": float(female_accept_rate),
        "male_accept_rate": float(male_accept_rate),
        "rate_diff_female_minus_male": float(female_accept_rate - male_accept_rate),
        "chi2": float(chi2),
        "chi2_p": float(p_chi2),
        "z_stat": float(stat_z),
        "z_p": float(p_z),
        "ct": ct.to_dict(),
        "missing_counts": {k: int(v) for k, v in missing.items()},
    }

    def logit_stats(model, n_obs):
        coef = model.params["female"]
        se = model.bse["female"]
        p = model.pvalues["female"]
        or_val = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
        return {
            "n_obs": int(n_obs),
            "coef": float(coef),
            "se": float(se),
            "p": float(p),
            "odds_ratio": or_val,
            "or_ci_low": ci_low,
            "or_ci_high": ci_high,
        }

    if isinstance(logit_unadj, BinaryResultsWrapper):
        summary["logit_unadj"] = logit_stats(logit_unadj, int(logit_unadj.nobs))
    else:
        summary["logit_unadj_error"] = str(logit_unadj)

    if isinstance(logit_adj, BinaryResultsWrapper):
        summary["logit_adj"] = logit_stats(logit_adj, int(logit_adj.nobs))
    else:
        summary["logit_adj_error"] = str(logit_adj)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
