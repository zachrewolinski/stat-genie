import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: enrollment / teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    print("Head of data with derived columns:")
    print(df[["feature6", "feature7", "stratio", "feature14", "feature15", "avg_score"]].head())

    print("\nStudent-teacher ratio summary:")
    print(df["stratio"].describe())

    print("\nAverage score summary:")
    print(df["avg_score"].describe())

    # Pearson and Spearman correlations between class size (student-teacher ratio) and performance
    pearson_corr = df["stratio"].corr(df["avg_score"], method="pearson")
    spearman_corr = df["stratio"].corr(df["avg_score"], method="spearman")
    print(
        f"\nPearson correlation between student-teacher ratio and avg score: {pearson_corr:.4f}"
    )
    print(
        f"Spearman correlation between student-teacher ratio and avg score: {spearman_corr:.4f}"
    )

    # Compare average performance across quartiles of student-teacher ratio
    df["stratio_quartile"] = pd.qcut(df["stratio"], 4, labels=False)
    group_means = df.groupby("stratio_quartile")["avg_score"].mean()
    print("\nAverage scores by student-teacher ratio quartile (0=lowest, 3=highest):")
    print(group_means)


if __name__ == "__main__":
    main()
