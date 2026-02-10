import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social information use: chose majority or minority option vs undemonstrated option
    df["social_use"] = df["y"].isin([2, 3]).astype(int)

    # Age groups (in years): 4-6, 7-10, 11-14
    df["age_group"] = pd.cut(
        df["age"],
        bins=[4, 7, 11, 15],
        right=False,
        labels=["4-6", "7-10", "11-14"],
    )

    # Majority choice among those who followed any demonstration
    df_social = df[df["social_use"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Chi-square tests for social information use
    social_culture_table = pd.crosstab(df["culture"], df["social_use"])
    chi2_sc, p_sc, _, _ = chi2_contingency(social_culture_table)

    social_age_table = pd.crosstab(df["age_group"], df["social_use"])
    chi2_sa, p_sa, _, _ = chi2_contingency(social_age_table)

    # Chi-square tests for majority preference
    majority_culture_table = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_mc, p_mc, _, _ = chi2_contingency(majority_culture_table)

    majority_age_table = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    chi2_ma, p_ma, _, _ = chi2_contingency(majority_age_table)

    print("Chi-square tests for variation in social information use and majority preference")
    print("--------------------------------------------------------------------------")
    print(f"Social information use ~ culture:  chi2 = {chi2_sc:.3f}, p = {p_sc:.4g}")
    print(f"Social information use ~ age_group: chi2 = {chi2_sa:.3f}, p = {p_sa:.4g}")
    print(f"Majority preference ~ culture:      chi2 = {chi2_mc:.3f}, p = {p_mc:.4g}")
    print(f"Majority preference ~ age_group:   chi2 = {chi2_ma:.3f}, p = {p_ma:.4g}")


if __name__ == "__main__":
    main()

