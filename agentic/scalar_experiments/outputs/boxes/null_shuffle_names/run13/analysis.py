import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: choosing either majority or minority
    df["social_choice"] = df["majority_first"] != 1

    # Preference for majority cue among social choices
    social_df = df[df["social_choice"]].copy()
    social_df["majority_pref"] = social_df["majority_first"] == 2

    results = {}

    # Variation across cultures (sites) for social vs asocial choices
    cont_social_site = pd.crosstab(df["y"], df["social_choice"])
    chi2_s_site, p_s_site, dof_s_site, _ = chi2_contingency(cont_social_site)
    prop_social_by_site = df.groupby("y")["social_choice"].mean()
    results["social_site"] = {
        "chi2": float(chi2_s_site),
        "p": float(p_s_site),
        "dof": int(dof_s_site),
        "min_prop": float(prop_social_by_site.min()),
        "max_prop": float(prop_social_by_site.max()),
    }

    # Variation across cultures for majority vs minority preference
    cont_maj_site = pd.crosstab(social_df["y"], social_df["majority_pref"])
    chi2_m_site, p_m_site, dof_m_site, _ = chi2_contingency(cont_maj_site)
    prop_maj_by_site = social_df.groupby("y")["majority_pref"].mean()
    results["majority_site"] = {
        "chi2": float(chi2_m_site),
        "p": float(p_m_site),
        "dof": int(dof_m_site),
        "min_prop": float(prop_maj_by_site.min()),
        "max_prop": float(prop_maj_by_site.max()),
    }

    # Age groups: early (4-7), middle (8-10), late (11-14)
    bins = [3.5, 7.5, 10.5, 14.5]
    labels = ["early", "middle", "late"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    social_df["age_group"] = pd.cut(social_df["age"], bins=bins, labels=labels)

    cont_social_age = pd.crosstab(df["age_group"], df["social_choice"])
    chi2_s_age, p_s_age, dof_s_age, _ = chi2_contingency(cont_social_age)
    prop_social_by_age = df.groupby("age_group")["social_choice"].mean()
    results["social_age"] = {
        "chi2": float(chi2_s_age),
        "p": float(p_s_age),
        "dof": int(dof_s_age),
        "min_prop": float(prop_social_by_age.min()),
        "max_prop": float(prop_social_by_age.max()),
    }

    cont_maj_age = pd.crosstab(social_df["age_group"], social_df["majority_pref"])
    chi2_m_age, p_m_age, dof_m_age, _ = chi2_contingency(cont_maj_age)
    prop_maj_by_age = social_df.groupby("age_group")["majority_pref"].mean()
    results["majority_age"] = {
        "chi2": float(chi2_m_age),
        "p": float(p_m_age),
        "dof": int(dof_m_age),
        "min_prop": float(prop_maj_by_age.min()),
        "max_prop": float(prop_maj_by_age.max()),
    }

    # Print a concise summary for inspection
    for key, val in results.items():
        print(f"{key}:")
        for k, v in val.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()

