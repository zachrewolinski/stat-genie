import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    if n == 0 or min(r, k) == 1:
        return np.nan
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: choosing any demonstrated option (majority or minority)
    df["social"] = (df["y"] != 1).astype(int)

    # Preference for majority cues among those who used social information
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Define developmental stages as age groups
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    df_social["age_group"] = pd.cut(
        df_social["age"], bins=bins, labels=labels, include_lowest=True
    )

    print("Overall outcome proportions (y):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    print("Social information use (proportion choosing a demonstrated option):")
    print(df["social"].mean())
    print()

    print("Majority preference among social learners:")
    print(df_social["majority_choice"].mean())
    print()

    # Social reliance by culture
    tab_culture_social = pd.crosstab(df["culture"], df["social"])
    chi2_cs, p_cs, dof_cs, _ = chi2_contingency(tab_culture_social)
    v_cs = cramers_v(tab_culture_social)
    print("Social reliance by culture:")
    print(tab_culture_social)
    print(
        f"chi2={chi2_cs:.3f}, dof={dof_cs}, p={p_cs:.4g}, CramersV={v_cs:.3f}",
    )
    print()

    # Social reliance by age group
    tab_age_social = pd.crosstab(df["age_group"], df["social"])
    chi2_as, p_as, dof_as, _ = chi2_contingency(tab_age_social)
    v_as = cramers_v(tab_age_social)
    print("Social reliance by age group:")
    print(tab_age_social)
    print(
        f"chi2={chi2_as:.3f}, dof={dof_as}, p={p_as:.4g}, CramersV={v_as:.3f}",
    )
    print()

    # Majority preference by culture
    tab_culture_majority = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_cm, p_cm, dof_cm, _ = chi2_contingency(tab_culture_majority)
    v_cm = cramers_v(tab_culture_majority)
    print("Majority preference by culture:")
    print(tab_culture_majority)
    print(
        f"chi2={chi2_cm:.3f}, dof={dof_cm}, p={p_cm:.4g}, CramersV={v_cm:.3f}",
    )
    print()

    # Majority preference by age group
    tab_age_majority = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    chi2_am, p_am, dof_am, _ = chi2_contingency(tab_age_majority)
    v_am = cramers_v(tab_age_majority)
    print("Majority preference by age group:")
    print(tab_age_majority)
    print(
        f"chi2={chi2_am:.3f}, dof={dof_am}, p={p_am:.4g}, CramersV={v_am:.3f}",
    )
    print()

    # Simple summaries by culture and age group for interpretation
    print("Mean social reliance by culture:")
    print(df.groupby("culture")["social"].mean())
    print()

    print("Mean social reliance by age group:")
    print(df.groupby("age_group")["social"].mean())
    print()

    print("Majority preference (probability of choosing majority) by culture:")
    print(df_social.groupby("culture")["majority_choice"].mean())
    print()

    print("Majority preference by age group:")
    print(df_social.groupby("age_group")["majority_choice"].mean())
    print()


if __name__ == "__main__":
    main()

