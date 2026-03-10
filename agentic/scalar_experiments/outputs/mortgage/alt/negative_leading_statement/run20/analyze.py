import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Clean column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Use accept as outcome (1 accepted, 0 denied). Ensure binary.
    if "accept" not in df.columns:
        raise ValueError("accept column missing")

    # Basic counts
    n = len(df)
    n_female = df["female"].sum()
    n_male = n - n_female

    # Approval rates by gender
    approve_female = df.loc[df["female"] == 1, "accept"].mean()
    approve_male = df.loc[df["female"] == 0, "accept"].mean()

    # Two-proportion z-test (female vs male approval rates)
    # counts of approvals
    success = np.array([
        df.loc[df["female"] == 1, "accept"].sum(),
        df.loc[df["female"] == 0, "accept"].sum(),
    ])
    nobs = np.array([n_female, n_male])
    # handle edge cases
    if n_female == 0 or n_male == 0:
        z_stat = np.nan
        p_value_z = np.nan
    else:
        prop = success / nobs
        pooled = success.sum() / nobs.sum()
        se = np.sqrt(pooled * (1 - pooled) * (1 / nobs[0] + 1 / nobs[1]))
        z_stat = (prop[0] - prop[1]) / se
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Logistic regression: accept ~ female (unadjusted)
    model_unadj = smf.logit("accept ~ female", data=df).fit(disp=False)

    # Logistic regression with controls
    # Exclude redundant or outcome-related vars: deny (inverse of accept) and Unnamed: 0 index.
    base_controls = [
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
    # Optional additional control that may be downstream of approval
    extra_controls = [
        "denied_PMI",
    ]
    # Only use controls that exist
    base_controls = [c for c in base_controls if c in df.columns]
    extra_controls = [c for c in extra_controls if c in df.columns]
    formula_base = "accept ~ female" + (" + " + " + ".join(base_controls) if base_controls else "")
    formula_full = formula_base + (" + " + " + ".join(extra_controls) if extra_controls else "")

    model_adj_base = smf.logit(formula_base, data=df).fit(disp=False)
    model_adj_full = smf.logit(formula_full, data=df).fit(disp=False)

    # Extract female coefficient stats
    def coef_summary(model):
        coef = model.params["female"]
        se = model.bse["female"]
        p = model.pvalues["female"]
        oratio = float(np.exp(coef))
        # 95% CI for odds ratio
        ci_low, ci_high = np.exp(model.conf_int().loc["female"].values)
        return {
            "coef": float(coef),
            "se": float(se),
            "p": float(p),
            "odds_ratio": oratio,
            "or_ci_low": float(ci_low),
            "or_ci_high": float(ci_high),
        }

    res = {
        "n": int(n),
        "n_female": int(n_female),
        "n_male": int(n_male),
        "approve_rate_female": float(approve_female),
        "approve_rate_male": float(approve_male),
        "approval_rate_diff_female_minus_male": float(approve_female - approve_male),
        "two_prop_z": float(z_stat),
        "two_prop_p": float(p_value_z),
        "unadjusted": coef_summary(model_unadj),
        "adjusted_base": coef_summary(model_adj_base),
        "adjusted_full": coef_summary(model_adj_full),
        "formula_adjusted_base": formula_base,
        "formula_adjusted_full": formula_full,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
