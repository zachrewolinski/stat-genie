import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In this dataset, the metadata indicates:
    # - Column "age" encodes frequency of extramarital intercourse in the last year (0 = none, >0 = some).
    # - Column "religiousness" is actually the children indicator: "yes"/"no" for whether there are children.
    #
    # We derive the variables we need for the research question:
    #   - any_affair: binary indicator of at least one extramarital affair.
    #   - has_children: binary indicator of children in the marriage.
    df["any_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = (df["religiousness"].str.lower() == "yes").astype(int)

    # Basic group summaries
    group_means = df.groupby("has_children")["any_affair"].mean()
    group_counts = df.groupby("has_children")["any_affair"].agg(["sum", "count"])

    # 2x2 table for chi-square test: rows = has_children (0/1), cols = any_affair (0/1)
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print("Affair frequency coding summary (unique values in 'age'):")
    print(sorted(df["age"].unique()))
    print()

    print("Proportion with any affair by children status:")
    for has_children, mean_val in group_means.items():
        label = "has children" if has_children == 1 else "no children"
        print(f"  {label}: {mean_val:.3f}")
    print()

    print("Counts of individuals and affairs by children status:")
    print(group_counts)
    print()

    print("Chi-square test for association between children and any affair (2x2 table):")
    print("Contingency table (rows: has_children 0/1, cols: any_affair 0/1):")
    print(contingency)
    print(f"chi2 = {chi2:.3f}, p-value = {p_value:.4f}, dof = {dof}")


if __name__ == "__main__":
    main()

