import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_reliance"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["age_group"] = pd.cut(
        df["feature3"],
        bins=[3.5, 6.5, 9.5, 12.5, 14.5],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )
    df["site"] = df["feature5"].astype(int)

    # Overall summaries
    df_social = df[df["social_reliance"] == 1].copy()
    sr_overall = df["social_reliance"].mean()
    maj_overall = df_social["majority_choice"].mean()

    print(f"Overall social reliance (proportion choosing any demonstrated option): {sr_overall:.3f}")
    print(
        "Overall majority preference among social choosers "
        f"(proportion choosing majority vs minority option): {maj_overall:.3f}"
    )

    def chi2_summary(rowvar, colvar, name: str) -> None:
        table = pd.crosstab(rowvar, colvar)
        chi2, p, dof, expected = chi2_contingency(table)
        print(f"\nChi-square for {name}")
        print("Contingency table:")
        print(table)
        print(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p:.6g}")

    # Chi-square tests for variation across sites and age groups
    chi2_summary(df["social_reliance"], df["site"], "social_reliance x site")
    chi2_summary(df["social_reliance"], df["age_group"], "social_reliance x age_group")
    chi2_summary(
        df_social["majority_choice"], df_social["site"], "majority_choice x site"
    )
    chi2_summary(
        df_social["majority_choice"],
        df_social["age_group"],
        "majority_choice x age_group",
    )

    # Group-level means to gauge effect sizes
    print("\nMean social reliance by site (proportion):")
    print(df.groupby("site")["social_reliance"].mean())

    print("\nMean social reliance by age_group (proportion):")
    print(df.groupby("age_group")["social_reliance"].mean())

    print("\nMean majority preference by site among social choosers (proportion):")
    print(df_social.groupby("site")["majority_choice"].mean())

    print(
        "\nMean majority preference by age_group among social choosers (proportion):"
    )
    print(df_social.groupby("age_group")["majority_choice"].mean())


if __name__ == "__main__":
    main()

