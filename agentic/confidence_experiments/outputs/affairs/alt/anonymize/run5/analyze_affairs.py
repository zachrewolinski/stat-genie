import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Create key variables
    df["has_children"] = (df["feature6"] == "yes").astype(int)
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic descriptives
    group_desc = df.groupby("has_children")[["feature2", "any_affair"]].agg(
        ["mean", "std", "count"]
    )

    # 2x2 table for presence of affairs vs children
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Logistic regression: probability of any affair ~ children
    logit_model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    coef = logit_model.params["has_children"]
    p_val = logit_model.pvalues["has_children"]
    odds_ratio = np.exp(coef)
    ci_low, ci_high = np.exp(logit_model.conf_int().loc["has_children"])

    # Non-parametric test on frequency among all respondents
    mw_stat, mw_p = stats.mannwhitneyu(
        df.loc[df["has_children"] == 1, "feature2"],
        df.loc[df["has_children"] == 0, "feature2"],
        alternative="two-sided",
    )

    print("Descriptive statistics by has_children (0 = no, 1 = yes):")
    print(group_desc)
    print()

    print("Contingency table of has_children vs any_affair:")
    print(contingency)
    print(f"Chi-square test p-value: {p_chi2:.4g}")
    print()

    print("Logistic regression: any_affair ~ has_children")
    print(logit_model.summary())
    print(f"Odds ratio for has_children: {odds_ratio:.3f}")
    print(f"95% CI for odds ratio: [{ci_low:.3f}, {ci_high:.3f}]")
    print()

    print("Mann-Whitney U test on affair frequency (feature2) by children status:")
    print(f"U statistic: {mw_stat:.3f}, p-value: {mw_p:.4g}")


if __name__ == "__main__":
    main()

