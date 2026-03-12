import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicators for any affair and for presence of children
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["has_children"] = (df["children"] == "yes").astype(int)

    print("Sample size:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    print("\nMean affair count by children:")
    print(df.groupby("children")["affairs"].mean())

    print("\nProportion with any affair by children:")
    print(df.groupby("children")["any_affair"].mean())

    # Two-sample t-test on affair counts
    counts_yes = df.loc[df["has_children"] == 1, "affairs"]
    counts_no = df.loc[df["has_children"] == 0, "affairs"]
    t_stat, t_p = stats.ttest_ind(counts_yes, counts_no, equal_var=False)
    print("\nT-test on affair counts (children: yes vs no):")
    print("  t-statistic:", t_stat)
    print("  p-value    :", t_p)

    # Non-parametric Mann-Whitney U test on affair counts
    u_stat, u_p = stats.mannwhitneyu(counts_yes, counts_no, alternative="two-sided")
    print("\nMann-Whitney U test on affair counts (children: yes vs no):")
    print("  U-statistic:", u_stat)
    print("  p-value    :", u_p)

    # Logistic regression for any affair ~ has_children
    model_simple = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    print("\nLogistic regression: any_affair ~ has_children")
    print(model_simple.summary())
    params_simple = model_simple.params
    or_children_simple = float(np.exp(params_simple["has_children"]))
    print("\nOdds ratio for has_children (simple model):", or_children_simple)

    # Logistic regression controlling for potential confounders
    formula_full = (
        "any_affair ~ has_children + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("\nLogistic regression: full model")
    print(model_full.summary())
    params_full = model_full.params
    or_children_full = float(np.exp(params_full["has_children"]))
    print("\nOdds ratio for has_children (full model):", or_children_full)


if __name__ == "__main__":
    main()

