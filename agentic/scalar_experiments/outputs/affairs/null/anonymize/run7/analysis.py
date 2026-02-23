import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Encode children indicator: 1 = children in marriage, 0 = no children.
    df = df.copy()
    df["has_children"] = df["feature6"].map({"yes": 1, "no": 0})

    if df["has_children"].isna().any():
        raise ValueError(f"Unexpected values in feature6: {df['feature6'].unique()}")

    # Outcome: any extramarital intercourse in the past year.
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    n_total = len(df)
    counts_children = df["has_children"].value_counts().sort_index()

    print(f"Total observations: {n_total}")
    print("Counts by children (0 = no, 1 = yes):")
    print(counts_children.to_string())

    # Descriptive statistics for frequency of affairs.
    freq_summary = (
        df.groupby("has_children")["feature2"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )
    print("\nAffair frequency (feature2) by children status:")
    print(freq_summary.to_string(float_format=lambda x: f"{x:0.3f}"))

    # Proportion with any affair by children status.
    prop_affair = (
        df.groupby("has_children")["affair_any"]
        .mean()
        .rename(index={0: "no_children", 1: "children"})
    )
    print("\nProportion with any affair by children status:")
    print(prop_affair.to_string(float_format=lambda x: f"{x:0.3f}"))

    # Welch t-test on affair frequency (treating feature2 as approximately continuous/ordinal).
    freq_children = df.loc[df["has_children"] == 1, "feature2"]
    freq_no_children = df.loc[df["has_children"] == 0, "feature2"]
    t_stat, t_p = stats.ttest_ind(freq_children, freq_no_children, equal_var=False)
    print(
        "\nWelch t-test for difference in affair frequency (children vs no children): "
        f"t = {t_stat:0.3f}, p = {t_p:0.4g}"
    )

    # Nonparametric check: Mann–Whitney U test on affair frequency.
    u_stat, u_p = stats.mannwhitneyu(
        freq_children,
        freq_no_children,
        alternative="two-sided",
    )
    print(
        "Mann–Whitney U test on affair frequency: "
        f"U = {u_stat:0.3f}, p = {u_p:0.4g}"
    )

    # Chi-square test of independence: children status vs any affair.
    contingency = pd.crosstab(df["has_children"], df["affair_any"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    print("\nContingency table (row = has_children, col = affair_any):")
    print(contingency.to_string())
    print(
        f"Chi-square test: chi2 = {chi2:0.3f}, dof = {dof}, p = {chi_p:0.4g}"
    )

    # Logistic regression: P(any affair) as a function of children (unadjusted).
    X = sm.add_constant(df["has_children"])
    y = df["affair_any"]
    logit_model = sm.Logit(y, X).fit(disp=False)

    coef_children = float(logit_model.params["has_children"])
    p_children = float(logit_model.pvalues["has_children"])
    odds_ratio = float(np.exp(coef_children))

    print("\nLogistic regression: affair_any ~ has_children")
    print(f"coef_children = {coef_children:0.3f}")
    print(f"odds_ratio   = {odds_ratio:0.3f}")
    print(f"p_value      = {p_children:0.4g}")

    # Predicted probabilities with and without children.
    X_pred = pd.DataFrame(
        {"const": [1.0, 1.0], "has_children": [0.0, 1.0]}
    )
    probs = logit_model.predict(X_pred)
    prob_no_children = float(probs.iloc[0])
    prob_children = float(probs.iloc[1])

    print(
        f"\nPredicted P(any affair) when no children: {prob_no_children:0.3f}"
    )
    print(
        f"Predicted P(any affair) when children:    {prob_children:0.3f}"
    )
    print(
        f"Difference (children - no children):      "
        f"{prob_children - prob_no_children:0.3f}"
    )


if __name__ == "__main__":
    main()

