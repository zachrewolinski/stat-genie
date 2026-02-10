import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Map outcome labels for clarity
    outcome_map = {1: "unchosen", 2: "majority", 3: "minority"}
    df["y_label"] = df["y"].map(outcome_map)

    n = len(df)

    # Overall reliance on majority vs others
    majority_share = (df["y"] == 2).mean()
    non_majority_share = 1 - majority_share

    # Variation across cultures
    culture_majority = df.groupby("culture")["y"].apply(lambda s: (s == 2).mean())
    culture_range = culture_majority.max() - culture_majority.min()

    # Variation across age (treat age as continuous)
    # Compute majority choice rate by age and its spread
    age_majority = df.groupby("age")["y"].apply(lambda s: (s == 2).mean())
    age_range = age_majority.max() - age_majority.min()

    # Simple measure of association: correlation between age and majority choice (binary)
    df["is_majority"] = (df["y"] == 2).astype(int)
    age_corr = df["age"].corr(df["is_majority"])

    # Summaries that we will inspect in the shell
    print("N:", n)
    print("Overall majority share:", majority_share)
    print("Non-majority share:", non_majority_share)
    print("\nMajority share by culture:")
    print(culture_majority.sort_index())
    print("Range across cultures:", culture_range)
    print("\nMajority share by age:")
    print(age_majority.sort_index())
    print("Range across ages:", age_range)
    print("\nCorrelation age vs majority (is_majority):", age_corr)


if __name__ == "__main__":
    main()

