import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator: had any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group statistics by children status
    group = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_with_affair=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .sort_index()
    )

    # Two-sample tests: difference in mean affair counts and in proportions
    yes_mask = df["children"] == "yes"
    no_mask = df["children"] == "no"

    affairs_yes = df.loc[yes_mask, "affairs"]
    affairs_no = df.loc[no_mask, "affairs"]

    # Welch's t-test for difference in means
    t_mean, p_mean = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)

    # Proportion test for any affair
    from statsmodels.stats.proportion import proportions_ztest

    counts = np.array(
        [df.loc[yes_mask, "has_affair"].sum(), df.loc[no_mask, "has_affair"].sum()]
    )
    nobs = np.array([yes_mask.sum(), no_mask.sum()])
    z_prop, p_prop = proportions_ztest(counts, nobs)

    # Logistic regression: children only
    logit_simple = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    coef_simple = logit_simple.params.get("C(children)[T.yes]", np.nan)
    p_simple = logit_simple.pvalues.get("C(children)[T.yes]", np.nan)
    or_simple = float(np.exp(coef_simple)) if np.isfinite(coef_simple) else np.nan

    # Logistic regression with controls
    formula_full = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    coef_full = logit_full.params.get("C(children)[T.yes]", np.nan)
    p_full = logit_full.pvalues.get("C(children)[T.yes]", np.nan)
    or_full = float(np.exp(coef_full)) if np.isfinite(coef_full) else np.nan

    # Collect key results in a JSON file for later inspection (not the final conclusion)
    results = {
        "group_stats": group.reset_index().to_dict(orient="records"),
        "t_test_means": {"t_stat": float(t_mean), "p_value": float(p_mean)},
        "prop_test": {"z_stat": float(z_prop), "p_value": float(p_prop)},
        "logit_simple": {
            "coef_children_yes": float(coef_simple),
            "p_value_children_yes": float(p_simple),
            "odds_ratio_children_yes": or_simple,
        },
        "logit_full": {
            "coef_children_yes": float(coef_full),
            "p_value_children_yes": float(p_full),
            "odds_ratio_children_yes": or_full,
        },
    }

    with Path("analysis_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Also print a concise summary to stdout for human inspection.
    print("Group stats by children:")
    print(group)
    print("\nWelch t-test (mean affair counts, yes - no):")
    print(f"t = {t_mean:.3f}, p = {p_mean:.4f}")
    print("\nProportion test (any affair, yes vs no):")
    print(f"z = {z_prop:.3f}, p = {p_prop:.4f}")
    print("\nLogit (children only) - coefficient for children=yes:")
    print(f"coef = {coef_simple:.3f}, p = {p_simple:.4f}, OR = {or_simple:.3f}")
    print("\nLogit (full controls) - coefficient for children=yes:")
    print(f"coef = {coef_full:.3f}, p = {p_full:.4f}, OR = {or_full:.3f}")


if __name__ == "__main__":
    main()

