import pandas as pd
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_info"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    demonstrated_mask = df["y"].isin([2, 3])
    df_demo = df[demonstrated_mask].copy()
    df_demo["majority_vs_minority"] = (df_demo["y"] == 2).astype(int)

    # Age groups to approximate developmental stages
    bins = [3.5, 6.5, 9.5, 11.5, 14.5]
    labels = ["4-6", "7-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    df_demo["age_group"] = pd.cut(df_demo["age"], bins=bins, labels=labels, right=True)

    print("=== Basic counts ===")
    print("N total:", len(df))
    print("Outcome distribution (y):")
    print(df["y"].value_counts().sort_index())
    print()

    print("Proportion using any social information (y in {2,3}):")
    print(df["social_info"].mean())
    print("Proportion choosing majority option (y==2):")
    print(df["majority_choice"].mean())
    print()

    # Helper to run chi-square and print summaries
    def chi_square_report(var_name: str, by: str, data: pd.DataFrame) -> None:
        print(f"=== Chi-square test: {var_name} by {by} ===")
        ctab = pd.crosstab(data[by], data[var_name])
        print("Contingency table:")
        print(ctab)
        chi2, p, dof, _ = stats.chi2_contingency(ctab)
        print(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p:.5f}")
        prop = ctab.div(ctab.sum(axis=1), axis=0)
        print("Row-wise proportions:")
        print(prop)
        print()

    # Social information reliance across cultures and age groups
    chi_square_report("social_info", "culture", df)
    chi_square_report("social_info", "age_group", df)

    # Majority choice across cultures and age groups (overall)
    chi_square_report("majority_choice", "culture", df)
    chi_square_report("majority_choice", "age_group", df)

    # Majority vs minority among children who chose a demonstrated option
    chi_square_report("majority_vs_minority", "culture", df_demo)
    chi_square_report("majority_vs_minority", "age_group", df_demo)


if __name__ == "__main__":
    main()

