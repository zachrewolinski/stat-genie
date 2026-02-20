import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # In this dataset, the metadata indicate that:
    # - Column "age" actually codes frequency of extramarital intercourse in the past year (0, 1, 2, 3, 7, 12, 12, 12).
    # - Column "religiousness" is a yes/no factor: "Are there children in the marriage?"
    # We will therefore:
    #   * Treat "age" as an ordinal measure of affair frequency.
    #   * Derive a binary indicator for any affair vs none.
    #   * Use "religiousness" to define presence of children.

    # Affair frequency and binary indicator
    df["affair_freq"] = df["age"]
    df["had_affair"] = (df["affair_freq"] > 0).astype(int)

    # Children indicator: 1 = children present, 0 = no children
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows where has_children is missing or malformed
    df = df.dropna(subset=["has_children"])

    # Basic group summaries
    group_summary = (
        df.groupby("has_children")
        .agg(
            mean_affair_freq=("affair_freq", "mean"),
            median_affair_freq=("affair_freq", "median"),
            prop_any_affair=("had_affair", "mean"),
            n=("had_affair", "size"),
        )
        .rename(index={0: "no_children", 1: "has_children"})
    )

    print("=== Group summary by children status ===")
    print(group_summary)
    print()

    # Two-sample t-test on affair frequency (Welch)
    freq_children = df.loc[df["has_children"] == 1, "affair_freq"]
    freq_no_children = df.loc[df["has_children"] == 0, "affair_freq"]
    t_stat, p_ttest = stats.ttest_ind(freq_children, freq_no_children, equal_var=False)

    print("=== Welch t-test: affair_freq by children status ===")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_ttest:.4g}")
    print()

    # Chi-squared test on any affair vs children
    ct = pd.crosstab(df["has_children"], df["had_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

    print("=== Chi-squared test: had_affair vs children status ===")
    print("Contingency table (rows: has_children [0/1], cols: had_affair [0/1]):")
    print(ct)
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_chi2:.4g}")
    print()

    # Logistic regression for any affair, adjusting for several covariates.
    # We use numeric covariates available in the data; note that names and
    # labels in the CSV are somewhat permuted relative to the original study,
    # but they still capture age, years married, education, etc.
    formula = "had_affair ~ has_children + occupation + children + rating + yearsmarried + C(gender)"

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("=== Logistic regression: had_affair ~ has_children + covariates ===")
    print(logit_model.summary())
    print()

    if "has_children" in logit_model.params.index:
        coef = logit_model.params["has_children"]
        se = logit_model.bse["has_children"]
        z_value = coef / se if se != 0 else np.nan
        p_value = logit_model.pvalues["has_children"]
        or_value = float(np.exp(coef))
        ci_low, ci_high = np.exp(logit_model.conf_int().loc["has_children"])

        print("Effect of having children (logistic model, outcome = any affair):")
        print(f"  log-odds coefficient = {coef:.3f}")
        print(f"  odds ratio           = {or_value:.3f}")
        print(f"  95% CI for OR        = ({ci_low:.3f}, {ci_high:.3f})")
        print(f"  z-value              = {z_value:.3f}")
        print(f"  p-value              = {p_value:.4g}")


if __name__ == "__main__":
    main()

