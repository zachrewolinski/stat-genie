import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    summary = []
    for has_children, group in df.groupby("children"):
        summary.append(
            {
                "children": has_children,
                "n": len(group),
                "mean_affairs": group["affairs"].mean(),
                "median_affairs": group["affairs"].median(),
                "prop_any_affair": group["any_affair"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary)
    print("Descriptive statistics by children status:")
    print(summary_df.to_string(index=False))

    # 2x2 table and chi-square test
    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    print("\nContingency table (children x any_affair):")
    print(contingency)
    print(f"\nChi-square test p-value: {p_chi2:.4g}")

    # Difference in mean affair counts by children status
    counts_no = df.loc[df["children"] == "no", "affairs"]
    counts_yes = df.loc[df["children"] == "yes", "affairs"]
    t_stat, p_ttest = stats.ttest_ind(counts_no, counts_yes, equal_var=False)
    print(
        f"\nT-test for mean affairs (no children vs. children): "
        f"t = {t_stat:.3f}, p-value = {p_ttest:.4g}"
    )

    # Logistic regression: probability of any affair ~ children (unadjusted)
    logit_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("\nUnadjusted logistic regression (any_affair ~ C(children)):")
    print(logit_unadj.summary())

    or_unadj = np.exp(logit_unadj.params)
    ci_unadj = np.exp(logit_unadj.conf_int())
    print("\nUnadjusted odds ratios:")
    for param, or_val in or_unadj.items():
        ci_low, ci_high = ci_unadj.loc[param]
        print(f"  {param}: OR = {or_val:.3f} (95% CI: {ci_low:.3f}, {ci_high:.3f})")

    # Logistic regression with covariate adjustment
    formula_adj = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print("\nAdjusted logistic regression:")
    print(logit_adj.summary())

    or_adj = np.exp(logit_adj.params)
    ci_adj = np.exp(logit_adj.conf_int())
    print("\nAdjusted odds ratios:")
    for param, or_val in or_adj.items():
        ci_low, ci_high = ci_adj.loc[param]
        print(f"  {param}: OR = {or_val:.3f} (95% CI: {ci_low:.3f}, {ci_high:.3f})")


if __name__ == "__main__":
    main()

