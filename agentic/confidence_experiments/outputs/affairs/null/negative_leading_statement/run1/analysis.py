import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create a binary indicator for any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children
    print("=== Descriptive statistics by children ===")
    group = df.groupby("children")
    summary = group["affairs"].agg(["mean", "std", "median", "count"])
    any_affair_rate = group["any_affair"].mean()
    print(summary)
    print("\nProportion with any affair by children:")
    print(any_affair_rate)

    # Two-sample t-test on numeric affairs score
    print("\n=== Two-sample t-test (affairs score by children) ===")
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    t_res = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    print(f"t-statistic = {t_res.statistic:.3f}, p-value = {t_res.pvalue:.4g}")

    # Nonparametric Mann-Whitney U test
    print("\n=== Mann-Whitney U test (affairs score by children) ===")
    u_res = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    print(f"U statistic = {u_res.statistic:.3f}, p-value = {u_res.pvalue:.4g}")

    # Chi-square test on any_affair vs children
    print("\n=== Chi-square test (any_affair vs children) ===")
    contingency = pd.crosstab(df["children"], df["any_affair"])
    print("Contingency table:")
    print(contingency)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    print(f"chi2 = {chi2:.3f}, p-value = {p_chi2:.4g}, dof = {dof}")

    # Logistic regression for any_affair with children and covariates
    print("\n=== Logistic regression: any_affair ~ children + covariates ===")
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print(logit_model.summary())

    # Extract and print the children effect explicitly
    params = logit_model.params
    conf = logit_model.conf_int()
    print("\nChildren effect (logistic regression, reference = children=='no'):")
    for name in params.index:
        if "C(children)" in name:
            est = params[name]
            ci_low, ci_high = conf.loc[name]
            pval = logit_model.pvalues[name]
            odds_ratio = np.exp(est)
            ci_low_or, ci_high_or = np.exp(ci_low), np.exp(ci_high)
            print(
                f"{name}: log-odds = {est:.3f} "
                f"(95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
                f"OR = {odds_ratio:.3f} "
                f"(95% CI [{ci_low_or:.3f}, {ci_high_or:.3f}]), "
                f"p-value = {pval:.4g}"
            )


if __name__ == "__main__":
    main()

