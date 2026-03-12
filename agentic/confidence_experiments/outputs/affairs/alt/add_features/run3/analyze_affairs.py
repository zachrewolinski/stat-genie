import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    # Binary indicator for having children
    df["has_children"] = df["children"].str.lower().eq("yes").astype(int)

    # Descriptive statistics by children status
    group_stats = (
        df.groupby("has_children")
        .agg(
            mean_affairs=("affairs", "mean"),
            std_affairs=("affairs", "std"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .rename(index={0: "no_children", 1: "has_children"})
    )

    print("Descriptive statistics by children status:")
    print(group_stats)
    print()

    # Non‑parametric comparison of the number of affairs
    affairs_no_children = df.loc[df["has_children"] == 0, "affairs"]
    affairs_with_children = df.loc[df["has_children"] == 1, "affairs"]
    u_stat, p_u = stats.mannwhitneyu(
        affairs_no_children, affairs_with_children, alternative="two-sided"
    )
    print("Mann–Whitney U test for affairs counts (no children vs. has children):")
    print(f"U statistic = {u_stat:.3f}, p-value = {p_u:.4g}")
    print()

    # Association between children and any affair (2x2 table and chi-squared test)
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    print("Contingency table: has_children (rows) x any_affair (columns):")
    print(contingency)
    print()
    print("Chi-squared test of independence:")
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_chi2:.4g}")
    print()

    # Logistic regression for probability of any affair, controlling for key covariates
    formula = (
        "any_affair ~ has_children + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression: any_affair ~ has_children + controls")
    print(logit_model.summary())
    print()

    coef_children = logit_model.params["has_children"]
    se_children = logit_model.bse["has_children"]
    p_children = logit_model.pvalues["has_children"]
    or_children = float(np.exp(coef_children))
    ci_low = float(np.exp(coef_children - 1.96 * se_children))
    ci_high = float(np.exp(coef_children + 1.96 * se_children))

    print("Effect of having children on odds of any affair:")
    print(f"log-odds coefficient = {coef_children:.3f}")
    print(f"odds ratio = {or_children:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"p-value = {p_children:.4g}")
    print()


if __name__ == "__main__":
    main()

