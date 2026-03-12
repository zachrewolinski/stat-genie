import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variable: any affair vs none
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("=== Sample size by children ===")
    print(df["children"].value_counts(dropna=False))
    print()

    print("=== Mean number of affairs by children ===")
    print(df.groupby("children")["affairs"].mean())
    print()

    print("=== Proportion with any affair by children ===")
    print(df.groupby("children")["any_affair"].mean())
    print()

    # Two-sample t-test for mean number of affairs by children status
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    t_res = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    print("=== Welch t-test: mean affairs, children yes vs no ===")
    print(f"t-statistic = {t_res.statistic:.4f}, p-value = {t_res.pvalue:.4g}")
    print(
        f"mean_yes = {affairs_yes.mean():.4f}, mean_no = {affairs_no.mean():.4f}, "
        f"diff_yes_minus_no = {affairs_yes.mean() - affairs_no.mean():.4f}"
    )
    print()

    # Logistic regression for any affair, adjusting for covariates
    formula_logit = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula_logit, data=df).fit(disp=False)
    print("=== Logistic regression for any affair ===")
    print(logit_model.summary())
    print()

    # Extract children effect from logistic regression
    params = logit_model.params
    conf_int = logit_model.conf_int()

    # Coefficient name for children=yes relative to no
    coef_name = "C(children)[T.yes]"
    if coef_name in params.index:
        coef = params[coef_name]
        se = logit_model.bse[coef_name]
        pvalue = logit_model.pvalues[coef_name]
        ci_low, ci_high = conf_int.loc[coef_name]
        odds_ratio = np.exp(coef)
        or_ci_low, or_ci_high = np.exp(ci_low), np.exp(ci_high)

        print("=== Children effect (any affair, logistic regression) ===")
        print(f"coef_logit = {coef:.4f}, SE = {se:.4f}, p-value = {pvalue:.4g}")
        print(
            f"odds_ratio = {odds_ratio:.4f}, "
            f"95% CI = [{or_ci_low:.4f}, {or_ci_high:.4f}] "
            "(children = yes vs no)"
        )
        print()
    else:
        print("Children coefficient not found in logistic regression results.")
        print()


if __name__ == "__main__":
    main()

