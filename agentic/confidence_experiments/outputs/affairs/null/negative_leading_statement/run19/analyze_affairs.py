import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variable: any affair vs none
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc_affairs = df.groupby("children")["affairs"].agg(
        ["mean", "std", "median", "count"]
    )
    prop_affair = df.groupby("children")["has_affair"].mean()

    print("Descriptive statistics for affairs by children status:")
    print(desc_affairs)
    print("\nProportion with any affair by children status:")
    print(prop_affair)

    # Two-sample tests for the count of affairs
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    ttest_res = stats.ttest_ind(
        affairs_yes, affairs_no, equal_var=False, nan_policy="omit"
    )
    mwu_res = stats.mannwhitneyu(
        affairs_yes, affairs_no, alternative="two-sided"
    )

    print("\nWelch t-test (affairs, children yes vs no):")
    print(ttest_res)
    print("\nMann-Whitney U test (affairs, children yes vs no):")
    print(mwu_res)

    # Chi-square test and logistic regression for any affair
    ctab = pd.crosstab(df["children"], df["has_affair"])
    chi2_res = stats.chi2_contingency(ctab)

    print("\nContingency table of children vs any affair:")
    print(ctab)
    print("\nChi-square test for independence (children vs any affair):")
    print(f"chi2={chi2_res[0]:.4f}, p-value={chi2_res[1]:.4g}")

    # Logistic regression controlling for key covariates
    logit_formula = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

    print("\nLogistic regression for any affair ~ children + covariates:")
    print(logit_model.summary())

    # Extract key effect for children
    params = logit_model.params
    conf_int = logit_model.conf_int()
    odds_ratios = np.exp(params)

    print("\nOdds ratios with 95% CI:")
    for name in params.index:
        ci_low, ci_high = conf_int.loc[name]
        print(
            f"{name}: OR={odds_ratios[name]:.3f}, "
            f"95% CI=({np.exp(ci_low):.3f}, {np.exp(ci_high):.3f}), "
            f"p={logit_model.pvalues[name]:.4g}"
        )


if __name__ == "__main__":
    main()

