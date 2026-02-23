import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Ensure expected columns exist
    if "children" not in df.columns or "affairs" not in df.columns:
        raise ValueError("Expected 'children' and 'affairs' columns in affairs.csv")

    # Drop rows with missing key values, if any
    df = df.dropna(subset=["children", "affairs"]).copy()

    # Binary encoding for children: 1 = yes, 0 = no
    df["children_binary"] = df["children"].map({"yes": 1, "no": 0})

    # Basic group statistics
    grouped = df.groupby("children")["affairs"]
    group_stats = grouped.agg(["mean", "std", "count"])

    # Two-sample tests comparing affairs counts between groups
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    # Welch's t-test (does not assume equal variance)
    ttest_res = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)

    # Non-parametric Mann–Whitney U test
    mwu_res = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")

    # Logistic regression for probability of any affair (affairs > 0)
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    X = sm.add_constant(df[["children_binary"]])
    y = df["has_affair"]
    logit_model = sm.Logit(y, X).fit(disp=False)
    logit_coef = float(logit_model.params["children_binary"])
    logit_pval = float(logit_model.pvalues["children_binary"])
    logit_or = float(np.exp(logit_coef))

    # Linear regression treating affairs count as continuous
    ols_model = sm.OLS(df["affairs"], X).fit()
    ols_coef = float(ols_model.params["children_binary"])
    ols_pval = float(ols_model.pvalues["children_binary"])

    results = {
        "group_stats": group_stats.to_dict(orient="index"),
        "t_test": {
            "statistic": float(ttest_res.statistic),
            "pvalue": float(ttest_res.pvalue),
        },
        "mannwhitney": {
            "statistic": float(mwu_res.statistic),
            "pvalue": float(mwu_res.pvalue),
        },
        "logistic": {
            "coef_children": logit_coef,
            "pvalue_children": logit_pval,
            "odds_ratio_children": logit_or,
        },
        "ols": {
            "coef_children": ols_coef,
            "pvalue_children": ols_pval,
        },
    }

    # Print as JSON so it is easy to inspect
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

