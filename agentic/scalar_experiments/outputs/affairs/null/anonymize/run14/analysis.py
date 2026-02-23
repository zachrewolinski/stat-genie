import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome variables
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Descriptive statistics by children status
    group_counts = df.groupby("children")["any_affair"].agg(["count", "sum"])
    group_props = group_counts["sum"] / group_counts["count"]
    group_mean_freq = df.groupby("children")["feature2"].mean()

    with_children = df.loc[df["children"] == 1, "feature2"]
    without_children = df.loc[df["children"] == 0, "feature2"]

    # Welch t-test on affair frequency
    ttest_res = stats.ttest_ind(
        with_children,
        without_children,
        equal_var=False,
        nan_policy="omit",
    )

    # Logistic regression: any affair ~ children + covariates
    formula_logit = (
        "any_affair ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula_logit, data=df).fit(disp=False)
    coef_children_logit = logit_model.params["children"]
    p_children_logit = logit_model.pvalues["children"]
    odds_ratio_children = float(np.exp(coef_children_logit))

    # Linear model on frequency as a robustness check
    formula_ols = (
        "feature2 ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    ols_model = smf.ols(formula_ols, data=df).fit()
    coef_children_ols = ols_model.params["children"]
    p_children_ols = ols_model.pvalues["children"]

    # Pack key results into a JSON blob for easy inspection
    results = {
        "group_props_any_affair": {
            "children_0": float(group_props.get(0, np.nan)),
            "children_1": float(group_props.get(1, np.nan)),
        },
        "group_mean_frequency": {
            "children_0": float(group_mean_freq.get(0, np.nan)),
            "children_1": float(group_mean_freq.get(1, np.nan)),
        },
        "t_test_frequency": {
            "statistic": float(ttest_res.statistic),
            "p_value": float(ttest_res.pvalue),
        },
        "logit_children": {
            "coef": float(coef_children_logit),
            "p_value": float(p_children_logit),
            "odds_ratio": odds_ratio_children,
        },
        "ols_children": {
            "coef": float(coef_children_ols),
            "p_value": float(p_children_ols),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

