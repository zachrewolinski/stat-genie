import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics: prevalence of any affair by children status
    prevalence_table = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prevalence"})
    )

    # Difference in mean number of affairs by children status
    mean_affairs = df.groupby("children")["affairs"].mean()

    # Chi-squared test of independence between children and any_affair
    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Unadjusted logistic regression: any_affair ~ children
    logit_unadj = smf.logit("any_affair ~ children_yes", data=df).fit(disp=0)
    coef_child_unadj = logit_unadj.params["children_yes"]
    p_child_unadj = logit_unadj.pvalues["children_yes"]
    or_child_unadj = float(np.exp(coef_child_unadj))

    # Adjusted logistic regression with key covariates
    formula_adj = (
        "any_affair ~ children_yes + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=0)
    coef_child_adj = logit_adj.params["children_yes"]
    p_child_adj = logit_adj.pvalues["children_yes"]
    or_child_adj = float(np.exp(coef_child_adj))

    print("Prevalence of any affair by children status:")
    print(prevalence_table)
    print("\nMean number of affairs by children status:")
    print(mean_affairs)
    print("\nChi-squared test of independence (children vs any_affair):")
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_chi2:.5f}")
    print("\nUnadjusted logistic regression (any_affair ~ children_yes):")
    print(f"children_yes coef = {coef_child_unadj:.3f}, "
          f"OR = {or_child_unadj:.3f}, p-value = {p_child_unadj:.5f}")
    print("\nAdjusted logistic regression:")
    print(f"children_yes coef = {coef_child_adj:.3f}, "
          f"OR = {or_child_adj:.3f}, p-value = {p_child_adj:.5f}")


if __name__ == "__main__":
    main()

