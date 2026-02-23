import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Recode key variables
    df["has_children"] = df["feature6"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Sanity checks
    print("Rows:", len(df))
    print("Has children value counts:")
    print(df["has_children"].value_counts(dropna=False))
    print("\nAny affair by children (crosstab):")
    tab = pd.crosstab(df["has_children"], df["any_affair"])
    print(tab)

    # Affair frequency by children status
    group_stats = df.groupby("has_children")["feature2"].agg(["mean", "median", "std", "count"])
    print("\nAffair frequency (feature2) by children status:")
    print(group_stats)

    # Two-sample t-test (Welch) on affair frequency
    with_children = df.loc[df["has_children"] == 1, "feature2"]
    without_children = df.loc[df["has_children"] == 0, "feature2"]
    ttest_res = stats.ttest_ind(without_children, with_children, equal_var=False)
    print("\nWelch t-test on affair frequency (no children vs children):")
    print(ttest_res)

    # Chi-square test on any affair vs children
    chi2, p_chi2, dof, expected = stats.chi2_contingency(tab)
    print("\nChi-square test of independence (children vs any affair):")
    print("chi2 =", chi2, "p-value =", p_chi2, "dof =", dof)
    print("Expected counts:")
    print(expected)

    # Logistic regression for any affair, controlling for covariates
    # feature3: gender (categorical)
    # feature4: age (numeric)
    # feature5: years married (numeric)
    # feature7: religiousness (numeric)
    # feature8: education (numeric)
    # feature9: occupation (numeric)
    # feature10: marriage rating (numeric)
    formula = (
        "any_affair ~ has_children + C(feature3) + "
        "feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    print("\nLogistic regression results for any_affair:")
    print(logit_model.summary())

    # Effect of children on odds of any affair
    coef_children = logit_model.params.get("has_children", np.nan)
    p_children = logit_model.pvalues.get("has_children", np.nan)
    if not np.isnan(coef_children):
        odds_ratio_children = float(np.exp(coef_children))
    else:
        odds_ratio_children = np.nan

    print("\nEffect of having children on any affair (logit model):")
    print("Coefficient (log-odds) for has_children:", coef_children)
    print("Odds ratio for has_children:", odds_ratio_children)
    print("p-value for has_children:", p_children)


if __name__ == "__main__":
    main()

