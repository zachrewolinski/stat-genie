import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent
    data_path = base_dir / "affairs.csv"

    df = pd.read_csv(data_path)

    # According to info.json:
    # - column "age" encodes frequency of extramarital intercourse in past year
    # - column "religiousness" actually answers "Are there children in the marriage?" (yes/no)
    df = df.copy()
    df["affair_freq"] = df["age"]
    df["has_children"] = (df["religiousness"] == "yes").astype(int)
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)

    print("Basic counts")
    print(df["has_children"].value_counts().rename(index={0: "no_children", 1: "has_children"}))
    print()

    group_stats = (
        df.groupby("has_children")
        .agg(
            n=("affair_freq", "size"),
            mean_affair_freq=("affair_freq", "mean"),
            median_affair_freq=("affair_freq", "median"),
            prop_any_affair=("has_affair", "mean"),
        )
        .rename(index={0: "no_children", 1: "has_children"})
    )

    print("Group-level affair statistics by children-in-marriage")
    print(group_stats)
    print()

    # Two-sample t-test on affair frequency between groups
    freq_with_children = df.loc[df["has_children"] == 1, "affair_freq"]
    freq_without_children = df.loc[df["has_children"] == 0, "affair_freq"]

    t_res = sm.stats.ttest_ind(freq_with_children, freq_without_children, usevar="unequal")
    t_stat, p_val, dfree = t_res
    print("T-test on affair frequency (with_children vs without_children)")
    print(f"t-statistic={t_stat:.3f}, df≈{dfree:.1f}, p-value={p_val:.4f}")
    print()

    # Logistic regression for any affair on children indicator only
    logit_simple = smf.logit("has_affair ~ has_children", data=df).fit(disp=False)
    print("Logistic regression: has_affair ~ has_children")
    print(logit_simple.summary())
    print()
    or_children = float(np.exp(logit_simple.params["has_children"]))
    print(f"Odds ratio for has_children: {or_children:.3f}")
    print()

    # Logistic regression with additional controls to check robustness.
    # Note: column names do not align with their semantic labels, but they
    # can still serve as generic controls.
    try:
        logit_controls = smf.logit(
            "has_affair ~ has_children + C(gender) + children + yearsmarried + rating + rownames",
            data=df,
        ).fit(disp=False)
        print("Logistic regression with controls:")
        print(logit_controls.summary())
        print()
        or_children_ctrl = float(np.exp(logit_controls.params["has_children"]))
        print(f"Controlled odds ratio for has_children: {or_children_ctrl:.3f}")
    except Exception as exc:  # pragma: no cover - defensive
        print("Logistic regression with controls failed:", exc)


if __name__ == "__main__":
    main()

