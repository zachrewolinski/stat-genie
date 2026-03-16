import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by presence of children
    group = df.groupby("children")
    summary = group[["affairs", "has_affair"]].agg(["mean", "std", "count"])
    print("=== Group summary by children ===")
    print(summary)
    print()

    # Difference in probability of having any affair (children yes vs no)
    table = pd.crosstab(df["children"], df["has_affair"])
    print("=== Contingency table: children x has_affair ===")
    print(table)
    print()

    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    print("Chi-square test for independence (children vs has_affair):")
    print(f"chi2 = {chi2:.3f}, p = {p_chi2:.4g}, dof = {dof}")
    print()

    # Compare mean count of affairs using a non-parametric test
    affairs_with_children = df.loc[df["children"] == "yes", "affairs"]
    affairs_without_children = df.loc[df["children"] == "no", "affairs"]
    u_stat, p_u = stats.mannwhitneyu(
        affairs_with_children, affairs_without_children, alternative="two-sided"
    )
    print("Mann-Whitney U test on affair counts (children yes vs no):")
    print(f"U = {u_stat:.3f}, p = {p_u:.4g}")
    print()

    # Logistic regression: probability of any affair, controlling for covariates
    # Treat children and gender as categorical; others as numeric covariates.
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("=== Logistic regression: has_affair on children and covariates ===")
    print(logit_model.summary())
    print()

    # Extract coefficient and p-value for children effect
    params = logit_model.params
    conf_int = logit_model.conf_int()
    pvalues = logit_model.pvalues

    # Children is coded with a reference level; identify the relevant term(s).
    # With C(children) and levels 'no' and 'yes', we expect a term like C(children)[T.yes].
    children_terms = [name for name in params.index if "C(children)" in name]
    print("Children terms in logistic model:")
    for term in children_terms:
        coef = params[term]
        pval = pvalues[term]
        ci_low, ci_high = conf_int.loc[term]
        odds_ratio = np.exp(coef)
        ci_low_or = np.exp(ci_low)
        ci_high_or = np.exp(ci_high)
        print(
            f"{term}: coef = {coef:.3f}, p = {pval:.4g}, "
            f"OR = {odds_ratio:.3f}, 95% CI OR = [{ci_low_or:.3f}, {ci_high_or:.3f}]"
        )


if __name__ == "__main__":
    main()

