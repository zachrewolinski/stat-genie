import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of having any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by presence of children
    group_means = (
        df.groupby("children")[["affairs", "has_affair"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("Group means by children (affairs and has_affair):")
    print(group_means)
    print()

    # Two-sample t-test for difference in probability of having any affair
    has_affair_yes = df.loc[df["children"] == "yes", "has_affair"]
    has_affair_no = df.loc[df["children"] == "no", "has_affair"]
    t_stat, p_val = stats.ttest_ind(
        has_affair_yes, has_affair_no, equal_var=False
    )
    print("T-test for difference in has_affair between children=yes and children=no:")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4g}")
    print()

    # Logistic regression for having any affair, controlling for covariates
    # children and gender are treated as categorical
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression for has_affair:")
    print(logit_model.summary())
    print()

    # Extract and print the coefficient for children=yes relative to children=no
    params = logit_model.params
    conf_int = logit_model.conf_int()
    # The exact parameter name depends on how pandas encodes categories;
    # we print all children-related coefficients for clarity.
    print("Children-related coefficients in logistic regression:")
    for name, value in params.items():
        if "C(children)" in name:
            ci_low, ci_high = conf_int.loc[name]
            odds_ratio = np.exp(value)
            ci_low_or = np.exp(ci_low)
            ci_high_or = np.exp(ci_high)
            print(
                f"  {name}: coef={value:.4f}, OR={odds_ratio:.4f}, "
                f"95% CI OR=({ci_low_or:.4f}, {ci_high_or:.4f})"
            )


if __name__ == "__main__":
    main()

