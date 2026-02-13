import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    print("=== Sample size ===")
    print(len(df))

    print("\n=== Mean number of affairs by children status ===")
    summary = df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"])
    print(summary)

    print("\n=== Proportion with any affair (affairs > 0) by children status ===")
    prop_any = df.groupby("children")["any_affair"].mean()
    print(prop_any)

    # Chi-square test for association between children and any_affair
    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    print("\n=== Chi-square test: children vs any_affair ===")
    print("Contingency table:")
    print(contingency)
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_chi2:.4g}")

    # Non-parametric comparison of affair counts
    kids = df.loc[df["children"] == "yes", "affairs"]
    no_kids = df.loc[df["children"] == "no", "affairs"]
    u_stat, p_u = stats.mannwhitneyu(kids, no_kids, alternative="two-sided")
    print("\n=== Mann-Whitney U test: affairs count by children status ===")
    print(f"U statistic = {u_stat:.3f}, p-value = {p_u:.4g}")
    print(f"Mean affairs (children = yes): {kids.mean():.3f}")
    print(f"Mean affairs (children = no): {no_kids.mean():.3f}")

    # Logistic regression for any_affair with children and key covariates
    formula = (
        "any_affair ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("\n=== Logistic regression: any_affair ~ children and covariates ===")
    print(logit_model.summary())

    children_coef = logit_model.params["children_yes"]
    children_p = logit_model.pvalues["children_yes"]
    children_or = np.exp(children_coef)
    ci_lower, ci_upper = logit_model.conf_int().loc["children_yes"]
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)

    print("\nEffect of having children (children_yes = 1 vs 0):")
    print(f"Log-odds coefficient = {children_coef:.3f}")
    print(f"Odds ratio = {children_or:.3f}")
    print(
        f"95% CI for odds ratio = ({or_ci_lower:.3f}, {or_ci_upper:.3f}), "
        f"p-value = {children_p:.4g}"
    )


if __name__ == "__main__":
    main()

