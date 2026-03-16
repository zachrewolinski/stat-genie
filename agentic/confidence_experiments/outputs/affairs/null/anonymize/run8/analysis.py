import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Outcome: frequency of extramarital intercourse in past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Main predictor: having children in the marriage (yes/no)
    df = df[df["feature6"].isin(["yes", "no"])].copy()
    df["children_binary"] = (df["feature6"] == "yes").astype(int)

    # Basic group summaries
    group = df.groupby("children_binary")
    mean_affairs = group["feature2"].mean()
    prop_any_affair = group["has_affair"].mean()
    count = group.size()

    # Non-parametric comparison of affair frequency by children status
    affairs_children = df.loc[df["children_binary"] == 1, "feature2"]
    affairs_no_children = df.loc[df["children_binary"] == 0, "feature2"]
    mw_stat, mw_p = stats.mannwhitneyu(
        affairs_children, affairs_no_children, alternative="two-sided"
    )

    # Logistic regression on "any affair", adjusting for key covariates
    # feature3: gender (factor)
    # feature4: age
    # feature5: years married
    # feature7: religiousness
    # feature8: education
    # feature9: occupation
    # feature10: self-rated marriage quality
    formula = (
        "has_affair ~ children_binary + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    children_coef = logit_model.params["children_binary"]
    children_p = logit_model.pvalues["children_binary"]
    conf_int = logit_model.conf_int().loc["children_binary"]
    or_children = float(np.exp(children_coef))
    or_ci_low = float(np.exp(conf_int[0]))
    or_ci_high = float(np.exp(conf_int[1]))

    results = {
        "n_total": int(df.shape[0]),
        "group_counts": {
            "children_yes": int(count.get(1, 0)),
            "children_no": int(count.get(0, 0)),
        },
        "mean_affairs": {
            "children_yes": float(mean_affairs.get(1, np.nan)),
            "children_no": float(mean_affairs.get(0, np.nan)),
        },
        "prop_any_affair": {
            "children_yes": float(prop_any_affair.get(1, np.nan)),
            "children_no": float(prop_any_affair.get(0, np.nan)),
        },
        "mannwhitney": {
            "statistic": float(mw_stat),
            "p_value": float(mw_p),
        },
        "logistic_children": {
            "coef": float(children_coef),
            "p_value": float(children_p),
            "odds_ratio": or_children,
            "ci_low": or_ci_low,
            "ci_high": or_ci_high,
        },
    }

    # Print as JSON so it is easy to inspect from the outside.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

