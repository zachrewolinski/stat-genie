import pandas as pd
import numpy as np
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["social_choice"] = df["feature1"].isin([2, 3]).astype(int)

    # Among children who followed a demonstrated option, 1 = majority, 0 = minority
    df_social = df[df["social_choice"] == 1].copy()
    df_social["majority_among_social"] = (df_social["feature1"] == 2).astype(int)

    # Age groups approximating developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["feature3"], bins=bins, labels=labels, right=True)
    df_social["age_group"] = pd.cut(
        df_social["feature3"], bins=bins, labels=labels, right=True
    )

    print("N observations:", len(df))
    print()

    # Overall proportions
    overall_majority = df["majority_choice"].mean()
    overall_social = df["social_choice"].mean()
    print(f"Overall majority choice rate: {overall_majority:.3f}")
    print(f"Overall social (any demonstrated) choice rate: {overall_social:.3f}")
    print()

    # Helper to describe proportions by a grouping variable
    def describe_by(group_col: str, var: str) -> None:
        print(f"Proportion of {var}=1 by {group_col}:")
        grouped = df.groupby(group_col)[var].agg(["mean", "count"])
        for idx, row in grouped.iterrows():
            print(f"  {group_col}={idx}: mean={row['mean']:.3f}, n={int(row['count'])}")
        print()

    describe_by("feature5", "social_choice")
    describe_by("feature5", "majority_choice")
    describe_by("age_group", "social_choice")
    describe_by("age_group", "majority_choice")

    # For majority vs minority among social choosers
    print("Proportion majority (vs minority) among social choosers by site:")
    grouped_social_site = df_social.groupby("feature5")["majority_among_social"].agg(
        ["mean", "count"]
    )
    for idx, row in grouped_social_site.iterrows():
        print(
            f"  site={idx}: majority rate={row['mean']:.3f}, n_social={int(row['count'])}"
        )
    print()

    print("Proportion majority (vs minority) among social choosers by age_group:")
    grouped_social_age = df_social.groupby("age_group")["majority_among_social"].agg(
        ["mean", "count"]
    )
    for idx, row in grouped_social_age.iterrows():
        print(
            f"  age_group={idx}: majority rate={row['mean']:.3f}, n_social={int(row['count'])}"
        )
    print()

    # Chi-square tests of independence
    def chi_square_test(row_var: str, col_var: str, data: pd.DataFrame) -> None:
        table = pd.crosstab(data[row_var], data[col_var])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        print(f"Chi-square test: {row_var} vs {col_var}")
        print("  chi2 =", f"{chi2:.3f}", "dof =", dof, "p-value =", f"{p:.5f}")
        print()

    print("=== Chi-square tests on reliance on social information (any demonstration) ===")
    chi_square_test("feature5", "social_choice", df)
    chi_square_test("age_group", "social_choice", df)

    print("=== Chi-square tests on choosing majority option (vs other choices) ===")
    chi_square_test("feature5", "majority_choice", df)
    chi_square_test("age_group", "majority_choice", df)

    print("=== Chi-square tests on majority vs minority among social choosers ===")
    chi_square_test("feature5", "majority_among_social", df_social)
    chi_square_test("age_group", "majority_among_social", df_social)


if __name__ == "__main__":
    main()

