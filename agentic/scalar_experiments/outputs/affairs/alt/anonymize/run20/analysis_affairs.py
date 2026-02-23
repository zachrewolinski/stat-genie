import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # feature2: frequency of extramarital intercourse in past year (ordinal coded numeric)
    # feature6: children in marriage (yes/no)
    df = df.copy()
    df["has_children"] = df["feature6"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Drop rows with missing key variables, if any
    df = df.dropna(subset=["has_children", "any_affair", "feature2"])

    # Group-wise descriptive stats
    group_stats = (
        df.groupby("has_children")
        .agg(
            mean_freq=("feature2", "mean"),
            median_freq=("feature2", "median"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )

    # Mann-Whitney U test on the ordinal frequency variable
    freq_children = df.loc[df["has_children"] == 1, "feature2"]
    freq_no_children = df.loc[df["has_children"] == 0, "feature2"]
    mw_stat, mw_p = stats.mannwhitneyu(
        freq_children, freq_no_children, alternative="two-sided"
    )

    # Chi-square test on binary any_affair vs children
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    # Simple logistic regression: any_affair ~ has_children
    logit_model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    odds_ratio = float(np.exp(logit_model.params["has_children"]))
    conf_int = logit_model.conf_int().loc["has_children"]
    or_ci_low = float(np.exp(conf_int[0]))
    or_ci_high = float(np.exp(conf_int[1]))
    p_value = float(logit_model.pvalues["has_children"])

    # Print a concise summary that we can inspect from the shell.
    print("Group statistics (has_children: 0=no, 1=yes):")
    print(group_stats.to_string(index=False))
    print("\nMann-Whitney U test on frequency feature2:")
    print(f"  U statistic = {mw_stat:.3f}, p-value = {mw_p:.5f}")
    print("\nChi-square test on any_affair vs has_children:")
    print(f"  chi2 = {chi2:.3f}, dof = {dof}, p-value = {chi2_p:.5f}")
    print("\nLogistic regression: any_affair ~ has_children")
    print(f"  has_children odds ratio = {odds_ratio:.3f}")
    print(
        f"  95% CI for OR = [{or_ci_low:.3f}, {or_ci_high:.3f}], "
        f"p-value = {p_value:.5f}"
    )


if __name__ == "__main__":
    main()

