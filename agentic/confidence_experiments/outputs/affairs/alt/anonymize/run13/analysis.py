import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Map variables using metadata:
    # feature2: frequency of extramarital intercourse in past year (0 = none)
    # feature6: children in the marriage? ("yes"/"no")
    affairs = df["feature2"]
    children_flag = df["feature6"] == "yes"

    # Binary indicator for having at least one extramarital affair
    any_affair = (affairs > 0).astype(int)
    has_children = children_flag.astype(int)

    # Descriptive statistics by children status
    desc = (
        df.assign(
            any_affair=any_affair,
            has_children=has_children,
        )
        .groupby("has_children")
        .agg(
            mean_affairs=("feature2", "mean"),
            std_affairs=("feature2", "std"),
            prop_any_affair=("any_affair", "mean"),
            n=("feature2", "size"),
        )
    )

    print("Descriptive statistics by children status (has_children=1 means children present):")
    print(desc)
    print()

    # Two-sample t-test for mean affair frequency between groups
    affairs_children = affairs[has_children == 1]
    affairs_no_children = affairs[has_children == 0]

    t_stat, p_ttest = stats.ttest_ind(
        affairs_children,
        affairs_no_children,
        equal_var=False,
    )

    print("Two-sample t-test (Welch) for mean affair frequency (children vs no children):")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_ttest:.4f}")
    print()

    # Logistic regression: probability of any affair as a function of children
    # Ensure a clear column name for the children indicator
    X_logit = sm.add_constant(pd.DataFrame({"has_children": has_children}))
    model_logit = sm.Logit(any_affair, X_logit)
    result_logit = model_logit.fit(disp=False)

    coef_children = result_logit.params["has_children"]
    p_children = result_logit.pvalues["has_children"]
    odds_ratio_children = float(np.exp(coef_children))

    print("Logistic regression: P(any affair) ~ has_children")
    print(result_logit.summary())
    print()
    print("Children coefficient details:")
    print(f"  Coefficient (log-odds) for has_children: {coef_children:.3f}")
    print(f"  Odds ratio for has_children: {odds_ratio_children:.3f}")
    print(f"  p-value for has_children: {p_children:.4f}")
    print()

    # Store core numerical results that might be useful when interpreting
    results = {
        "descriptive": desc.reset_index().to_dict(orient="list"),
        "ttest": {"t_stat": float(t_stat), "p_value": float(p_ttest)},
        "logit_has_children": {
            "coef": float(coef_children),
            "odds_ratio": odds_ratio_children,
            "p_value": float(p_children),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
