import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Based on the metadata in info.json:
    # - The "age" column actually encodes frequency of extramarital intercourse
    #   over the past year (0 = none, >0 = some affairs).
    # - The "religiousness" column is a yes/no factor answering
    #   "Are there children in the marriage?"
    df["has_affair"] = (df["age"] > 0).astype(int)
    df["children_in_marriage"] = (df["religiousness"] == "yes").astype(int)

    # Basic sanity checks
    print("Value counts for has_affair (0=no, 1=yes):")
    print(df["has_affair"].value_counts().sort_index(), end="\n\n")

    print("Value counts for children_in_marriage (0=no, 1=yes):")
    print(df["children_in_marriage"].value_counts().sort_index(), end="\n\n")

    # Cross-tabulation and difference in proportions
    ct = pd.crosstab(df["children_in_marriage"], df["has_affair"])
    ct.index = ["no_children", "children"]
    ct.columns = ["no_affair", "affair"]
    print("Contingency table: children_in_marriage x has_affair")
    print(ct, end="\n\n")

    # Proportions of having any affair by children status
    prop_affair = ct["affair"] / ct.sum(axis=1)
    print("Proportion with at least one affair:")
    print(prop_affair, end="\n\n")

    # Chi-square test of independence
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct.values)
    print(f"Chi-square test p-value: {p_chi2:.4g}", end="\n\n")

    # Unadjusted logistic regression: P(has_affair) ~ children_in_marriage
    model_unadj = smf.logit("has_affair ~ children_in_marriage", data=df).fit(disp=0)
    print("Unadjusted logistic regression results:")
    print(model_unadj.summary(), end="\n\n")

    beta = model_unadj.params["children_in_marriage"]
    se = model_unadj.bse["children_in_marriage"]
    odds_ratio = float(np.exp(beta))
    ci_low = float(np.exp(beta - 1.96 * se))
    ci_high = float(np.exp(beta + 1.96 * se))
    p_value = float(model_unadj.pvalues["children_in_marriage"])

    print(
        "Effect of having children on odds of any affair "
        "(children_in_marriage vs no_children):"
    )
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  p-value: {p_value:.4g}")

    # Adjusted logistic regression including key demographic and relationship covariates.
    # Here we control for:
    # - gender (categorical)
    # - occupation (proxy for age band, per metadata)
    # - children (years married)
    # - rating (religiousness scale)
    # - yearsmarried (continuous years married)
    formula_adj = (
        "has_affair ~ children_in_marriage + C(gender) + occupation + children "
        "+ rating + yearsmarried"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=0)
    print("Adjusted logistic regression results:")
    print(model_adj.summary(), end="\n\n")

    beta_adj = model_adj.params["children_in_marriage"]
    se_adj = model_adj.bse["children_in_marriage"]
    odds_ratio_adj = float(np.exp(beta_adj))
    ci_low_adj = float(np.exp(beta_adj - 1.96 * se_adj))
    ci_high_adj = float(np.exp(beta_adj + 1.96 * se_adj))
    p_value_adj = float(model_adj.pvalues["children_in_marriage"])

    print(
        "Adjusted effect of having children on odds of any affair "
        "(children_in_marriage vs no_children):"
    )
    print(f"  Odds ratio: {odds_ratio_adj:.3f}")
    print(f"  95% CI: [{ci_low_adj:.3f}, {ci_high_adj:.3f}]")
    print(f"  p-value: {p_value_adj:.4g}")


if __name__ == "__main__":
    main()
