import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = df["children"].astype(str).str.lower().eq("yes").astype(int)

    print("Total rows:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts(dropna=False))

    # Group-level descriptive statistics
    group_stats = df.groupby("children")["affairs"].agg(
        ["mean", "median", "std", "count"]
    )
    print("\nAffairs by children group (count outcome):")
    print(group_stats)

    # Difference in counts between groups
    no_children = df.loc[df["children"] == "no", "affairs"]
    yes_children = df.loc[df["children"] == "yes", "affairs"]

    ttest = stats.ttest_ind(no_children, yes_children, equal_var=False)
    print("\nWelch t-test on affairs counts (no children vs. children):")
    print(f"mean_no_children: {no_children.mean():.3f}")
    print(f"mean_yes_children: {yes_children.mean():.3f}")
    print(f"t_statistic: {ttest.statistic:.3f}, p_value: {ttest.pvalue:.6f}")

    # Logistic regression on any affair vs none, controlling for main covariates
    formula = (
        "has_affair ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    print("\nLogistic regression: has_affair on children + covariates")
    print(logit_model.summary())

    children_coef = float(logit_model.params["children_yes"])
    children_p = float(logit_model.pvalues["children_yes"])
    odds_ratio = float(np.exp(children_coef))

    print(
        "\nEffect of having children (children_yes == 1) "
        "on odds of any affair (has_affair == 1):"
    )
    print(f"children_yes coefficient: {children_coef:.4f}")
    print(f"children_yes p-value: {children_p:.6f}")
    print(f"children_yes odds ratio: {odds_ratio:.3f}")

    # Also report simple affair prevalence by group
    prevalence = df.groupby("children")["has_affair"].mean()
    print("\nPrevalence of any affair by children group:")
    print(prevalence)


if __name__ == "__main__":
    main()

