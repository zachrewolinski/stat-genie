import pandas as pd
import numpy as np
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json descriptions:
    # - Column "age" actually encodes frequency of extramarital intercourse in past year.
    #   0 = none, 1 = once, 2 = twice, 3 = 3 times, 7 = 4–10 times, 12 = monthly/weekly/daily.
    # - Column "religiousness" is a yes/no factor answering
    #   "Are there children in the marriage?"

    # Outcome: any extramarital affair in the past year
    df["affair_freq"] = df["age"]
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Exposure: presence of children in the marriage (1 = children present, 0 = no children)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop rows with missing mapping (if any)
    df = df.dropna(subset=["has_children"])
    df["has_children"] = df["has_children"].astype(int)

    # Basic group summaries
    grouped = df.groupby("has_children")

    summary = grouped["affair_freq"].agg(
        n="count", mean="mean", std="std"
    ).reset_index()

    # Proportion with any affair by children status
    prop_any = grouped["any_affair"].mean().reset_index(name="prop_any_affair")

    # Welch t-test on affair frequency between groups
    freq_children = df.loc[df["has_children"] == 1, "affair_freq"]
    freq_no_children = df.loc[df["has_children"] == 0, "affair_freq"]
    t_res = stats.ttest_ind(
        freq_children,
        freq_no_children,
        equal_var=False,
        nan_policy="omit",
    )

    # Chi-square test on any_affair vs has_children
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, chi_p, chi_dof, chi_expected = stats.chi2_contingency(contingency)

    # Print a compact summary for manual inspection
    print("=== Group summaries by has_children (1=yes, 0=no) ===")
    print(summary.to_string(index=False))
    print()
    print("=== Proportion with any affair by has_children ===")
    print(prop_any.to_string(index=False))
    print()
    print("=== Welch t-test on affair frequency ===")
    print(f"t-statistic = {t_res.statistic:.3f}, p-value = {t_res.pvalue:.5f}")
    print()
    print("=== Chi-square test on any_affair vs has_children ===")
    print(f"chi2 = {chi2:.3f}, df = {chi_dof}, p-value = {chi_p:.5f}")
    print()
    print("Contingency table (rows: has_children, cols: any_affair 0/1):")
    print(contingency.to_string())


if __name__ == "__main__":
    main()

