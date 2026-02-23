import json

import pandas as pd
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode outcomes
    df["social"] = (df["y"] != 1).astype(int)  # 1 if majority or minority option, 0 if undemonstrated
    df["choice_label"] = df["y"].map({1: "undemonstrated", 2: "majority", 3: "minority"})

    # Coarse developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print("Sample size:", len(df))
    print("\nOutcome distribution (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nSocial information reliance rate (any demonstrated option):")
    print(df["social"].mean())

    print("\nSocial reliance by culture (proportion choosing demonstrated option):")
    social_by_culture = df.groupby("culture")["social"].mean()
    print(social_by_culture)

    print("\nSocial reliance by age group (proportion choosing demonstrated option):")
    social_by_age = df.groupby("age_group")["social"].mean()
    print(social_by_age)

    print("\nMajority preference among social choosers by culture (proportion choosing majority):")
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()
    print(majority_by_culture)

    print("\nMajority preference among social choosers by age group (proportion choosing majority):")
    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()
    print(majority_by_age)

    # Chi-square tests for associations
    print("\nChi-square tests for associations:")

    ct_social_culture = pd.crosstab(df["culture"], df["social"])
    chi2_sc, p_sc, dof_sc, _ = stats.chi2_contingency(ct_social_culture)
    print(f"Social vs culture: chi2={chi2_sc:.2f}, dof={dof_sc}, p={p_sc:.4g}")

    ct_social_age = pd.crosstab(df["age_group"], df["social"])
    chi2_sa, p_sa, dof_sa, _ = stats.chi2_contingency(ct_social_age)
    print(f"Social vs age_group: chi2={chi2_sa:.2f}, dof={dof_sa}, p={p_sa:.4g}")

    ct_maj_culture = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_mc, p_mc, dof_mc, _ = stats.chi2_contingency(ct_maj_culture)
    print(f"Majority vs culture: chi2={chi2_mc:.2f}, dof={dof_mc}, p={p_mc:.4g}")

    ct_maj_age = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    chi2_ma, p_ma, dof_ma, _ = stats.chi2_contingency(ct_maj_age)
    print(f"Majority vs age_group: chi2={chi2_ma:.2f}, dof={dof_ma}, p={p_ma:.4g}")

    # Save the key statistics in a JSON file for easier inspection if needed.
    results = {
        "social_by_culture": social_by_culture.to_dict(),
        "social_by_age_group": {str(k): float(v) for k, v in social_by_age.to_dict().items()},
        "majority_by_culture": majority_by_culture.to_dict(),
        "majority_by_age_group": {str(k): float(v) for k, v in majority_by_age.to_dict().items()},
        "tests": {
            "social_vs_culture": {"chi2": chi2_sc, "dof": dof_sc, "p": p_sc},
            "social_vs_age_group": {"chi2": chi2_sa, "dof": dof_sa, "p": p_sa},
            "majority_vs_culture": {"chi2": chi2_mc, "dof": dof_mc, "p": p_mc},
            "majority_vs_age_group": {"chi2": chi2_ma, "dof": dof_ma, "p": p_ma},
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

