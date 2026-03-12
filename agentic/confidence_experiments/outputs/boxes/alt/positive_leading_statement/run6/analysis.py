import pandas as pd
import numpy as np
from scipy import stats


def chi2_with_props(table, label_positive=None):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    props = table.div(table.sum(axis=1), axis=0)
    if label_positive is not None and label_positive in props.columns:
        pos_props = props[label_positive]
    else:
        pos_props = None
    return chi2, p, dof, expected, props, pos_props


def main():
    df = pd.read_csv("boxes.csv")

    # Define key derived variables
    df["social"] = (df["y"] != 1).astype(int)  # 1 if followed any demonstrated option
    df["majority_choice"] = (df["y"] == 2).astype(int)  # 1 if chose majority option

    # Age groups for developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    social_df = df.copy()
    majority_df = df[df["social"] == 1].copy()

    print("N observations:", len(df))
    print()

    print("Overall outcome distribution (proportions):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print("Coding: 1 = undemonstrated, 2 = majority, 3 = minority")
    print()

    # Reliance on social information (any demonstrated option) by age group
    social_age_tab = pd.crosstab(social_df["age_group"], social_df["social"])
    social_age_chi2, social_age_p, social_age_dof, _, social_age_props, social_age_pos = chi2_with_props(
        social_age_tab, label_positive=1
    )

    print("Proportion choosing any demonstrated (social) option by age group:")
    print(social_age_pos.round(3))
    print(
        f"Chi-square test of social reliance vs age_group: "
        f"chi2={social_age_chi2:.3f}, dof={social_age_dof}, p={social_age_p:.4g}"
    )
    print()

    # Reliance on social information by culture
    social_cult_tab = pd.crosstab(social_df["culture"], social_df["social"])
    social_cult_chi2, social_cult_p, social_cult_dof, _, social_cult_props, social_cult_pos = chi2_with_props(
        social_cult_tab, label_positive=1
    )

    print("Proportion choosing any demonstrated (social) option by culture:")
    print(social_cult_pos.round(3))
    print(
        f"Chi-square test of social reliance vs culture: "
        f"chi2={social_cult_chi2:.3f}, dof={social_cult_dof}, p={social_cult_p:.4g}"
    )
    print()

    # Preference for majority (vs minority) among children who use social information
    maj_age_tab = pd.crosstab(majority_df["age_group"], majority_df["majority_choice"])
    maj_age_chi2, maj_age_p, maj_age_dof, _, maj_age_props, maj_age_pos = chi2_with_props(
        maj_age_tab, label_positive=1
    )

    print("Proportion choosing majority (vs minority) option among social choosers, by age group:")
    print(maj_age_pos.round(3))
    print(
        f"Chi-square test of majority preference vs age_group: "
        f"chi2={maj_age_chi2:.3f}, dof={maj_age_dof}, p={maj_age_p:.4g}"
    )
    print()

    maj_cult_tab = pd.crosstab(majority_df["culture"], majority_df["majority_choice"])
    maj_cult_chi2, maj_cult_p, maj_cult_dof, _, maj_cult_props, maj_cult_pos = chi2_with_props(
        maj_cult_tab, label_positive=1
    )

    print("Proportion choosing majority (vs minority) option among social choosers, by culture:")
    print(maj_cult_pos.round(3))
    print(
        f"Chi-square test of majority preference vs culture: "
        f"chi2={maj_cult_chi2:.3f}, dof={maj_cult_dof}, p={maj_cult_p:.4g}"
    )
    print()


if __name__ == "__main__":
    main()

