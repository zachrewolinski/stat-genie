import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Based on info.json metadata:
    # - Column "age" is actually the affair frequency variable
    # - Column "religiousness" encodes whether there are children in the marriage ("yes"/"no")
    df["affairs_freq"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Keep only rows with necessary fields present
    cols = [
        "affairs_freq",
        "has_children",
        "gender",
        "yearsmarried",
        "rating",
        "education",
        "occupation",
        "rownames",
    ]
    sub = df[cols].dropna().copy()

    # Basic distribution summaries
    print("Basic group statistics for affair frequency by children status")
    group_stats = (
        sub.groupby("has_children")["affairs_freq"]
        .agg(["mean", "median", "std", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )
    print(group_stats)
    print()

    # Two-sample tests for mean/central tendency differences
    children = sub.loc[sub["has_children"] == 1, "affairs_freq"]
    no_children = sub.loc[sub["has_children"] == 0, "affairs_freq"]

    t_stat, t_p = stats.ttest_ind(children, no_children, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(children, no_children, alternative="two-sided")

    print("Welch t-test (affair frequency, children vs no children)")
    print(f"  t = {t_stat:.3f}, p-value = {t_p:.4f}")
    print("Mann-Whitney U test (affair frequency, children vs no children)")
    print(f"  U = {u_stat:.3f}, p-value = {u_p:.4f}")
    print()

    # Binary outcome: any affair vs none
    sub["any_affair"] = (sub["affairs_freq"] > 0).astype(int)
    print("Proportion with any affair by children status")
    prop = sub.groupby("has_children")["any_affair"].mean().rename(
        index={0: "no_children", 1: "has_children"}
    )
    print(prop)
    print()

    # Unadjusted logistic regression
    logit_model = smf.logit("any_affair ~ has_children", data=sub).fit(disp=False)
    logit_coef = logit_model.params["has_children"]
    logit_p = logit_model.pvalues["has_children"]
    logit_or = float(np.exp(logit_coef))

    print("Unadjusted logistic regression: any_affair ~ has_children")
    print(f"  coef(has_children) = {logit_coef:.3f}")
    print(f"  odds ratio         = {logit_or:.3f}")
    print(f"  p-value            = {logit_p:.4f}")
    print()

    # Adjusted logistic regression with key covariates
    logit_adj = smf.logit(
        "any_affair ~ has_children + C(gender) + yearsmarried + rating + education + occupation + rownames",
        data=sub,
    ).fit(disp=False)
    logit_adj_coef = logit_adj.params["has_children"]
    logit_adj_p = logit_adj.pvalues["has_children"]
    logit_adj_or = float(np.exp(logit_adj_coef))

    print(
        "Adjusted logistic regression: any_affair ~ has_children + controls (gender, yearsmarried, rating, education, occupation, rownames)"
    )
    print(f"  coef(has_children) = {logit_adj_coef:.3f}")
    print(f"  odds ratio         = {logit_adj_or:.3f}")
    print(f"  p-value            = {logit_adj_p:.4f}")
    print()

    # Poisson regression for the count-like affair frequency
    poisson = smf.glm(
        "affairs_freq ~ has_children", data=sub, family=sm.families.Poisson()
    ).fit()
    pois_coef = poisson.params["has_children"]
    pois_p = poisson.pvalues["has_children"]
    pois_rr = float(np.exp(pois_coef))

    print("Poisson regression: affairs_freq ~ has_children")
    print(f"  coef(has_children) = {pois_coef:.3f}")
    print(f"  rate ratio         = {pois_rr:.3f}")
    print(f"  p-value            = {pois_p:.4f}")


if __name__ == "__main__":
    main()

