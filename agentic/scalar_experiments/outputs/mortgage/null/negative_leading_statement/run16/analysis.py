import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    df = pd.read_csv("mortgage.csv")

    # Ensure binary outcome
    if "deny" not in df.columns:
        raise ValueError("deny column not found")

    # Basic group stats
    grouped = df.groupby("female")["deny"].agg(["mean", "count", "sum"]).rename(columns={"sum": "denied"})

    # 2x2 contingency table for chi-square
    # rows: female 0/1, cols: deny 0/1
    contingency = pd.crosstab(df["female"], df["deny"]).reindex(index=[0,1], columns=[0,1])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency, correction=False)

    # Logistic regression with controls
    controls = [
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
    model_df = df[controls + ["deny"]].dropna()
    y = model_df["deny"].astype(float)
    X = model_df[controls]
    X = sm.add_constant(X, has_constant="add")

    logit = sm.Logit(y, X)
    result = logit.fit(disp=False, maxiter=200)

    coef = result.params["female"]
    se = result.bse["female"]
    z = coef / se
    p = result.pvalues["female"]
    odds_ratio = np.exp(coef)
    ci_low, ci_high = result.conf_int().loc["female"].values
    or_ci_low = np.exp(ci_low)
    or_ci_high = np.exp(ci_high)

    # Marginal effect at means for interpretability
    margeff = result.get_margeff(at="mean")
    me_frame = margeff.summary_frame()
    me = float(me_frame.loc["female", "dy/dx"])
    me_se = float(me_frame.loc["female", "Std. Err."])
    p_col = None
    for candidate in ["P>|z|", "Pr(>|z|)", "Pr(>z)", "P>|t|", "Pr(>|t|)"]:
        if candidate in me_frame.columns:
            p_col = candidate
            break
    if p_col is None:
        raise KeyError(f"Could not find p-value column in marginal effects frame: {list(me_frame.columns)}")
    me_p = float(me_frame.loc["female", p_col])

    output = {
        "grouped": grouped.to_dict(),
        "contingency": contingency.to_dict(),
        "chi2": float(chi2),
        "chi2_p": float(p_chi),
        "logit": {
            "coef": float(coef),
            "se": float(se),
            "z": float(z),
            "p": float(p),
            "odds_ratio": float(odds_ratio),
            "or_ci_low": float(or_ci_low),
            "or_ci_high": float(or_ci_high),
        },
        "marginal_effect": {
            "dy_dx": float(me),
            "se": float(me_se),
            "p": float(me_p),
        },
        "n": int(model_df.shape[0]),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
