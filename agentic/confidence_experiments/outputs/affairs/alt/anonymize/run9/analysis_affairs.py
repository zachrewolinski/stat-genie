import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome variables
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Children indicator
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    # Descriptive statistics by children status
    mean_freq = df.groupby("feature6")["feature2"].mean()
    prop_any = df.groupby("feature6")["any_affair"].mean()

    # Chi-square test for any affair vs children
    contingency = pd.crosstab(df["any_affair"], df["feature6"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Logistic regression: any affair ~ children (unadjusted)
    logit_unadj = smf.logit("any_affair ~ C(feature6)", data=df).fit(disp=False)

    # Logistic regression with basic covariate adjustment
    formula_adj = (
        "any_affair ~ C(feature6) + feature4 + feature5 + feature7 + "
        "feature8 + feature9 + feature10 + C(feature3)"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)

    coef_unadj = float(logit_unadj.params["C(feature6)[T.yes]"])
    p_unadj = float(logit_unadj.pvalues["C(feature6)[T.yes]"])
    or_unadj = float(np.exp(coef_unadj))
    ci_unadj = np.exp(logit_unadj.conf_int().loc["C(feature6)[T.yes]"].to_numpy())

    coef_adj = float(logit_adj.params["C(feature6)[T.yes]"])
    p_adj = float(logit_adj.pvalues["C(feature6)[T.yes]"])
    or_adj = float(np.exp(coef_adj))
    ci_adj = np.exp(logit_adj.conf_int().loc["C(feature6)[T.yes]"].to_numpy())

    results = {
        "mean_freq_by_children": mean_freq.to_dict(),
        "prop_any_affair_by_children": prop_any.to_dict(),
        "chi2_any_affair_children": {
            "chi2": float(chi2),
            "p_value": float(p_chi2),
            "dof": int(dof),
        },
        "logit_unadjusted_children": {
            "coef": coef_unadj,
            "odds_ratio": or_unadj,
            "ci_95": ci_unadj.tolist(),
            "p_value": p_unadj,
        },
        "logit_adjusted_children": {
            "coef": coef_adj,
            "odds_ratio": or_adj,
            "ci_95": ci_adj.tolist(),
            "p_value": p_adj,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

