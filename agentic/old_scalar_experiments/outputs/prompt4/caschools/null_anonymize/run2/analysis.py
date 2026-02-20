import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # student-teacher ratio: enrollment per teacher
    df["stratio"] = df["feature6"] / df["feature7"]

    # academic performance: average of reading and math scores
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Basic sanity checks
    print("Summary of student-teacher ratio (stratio):")
    print(df["stratio"].describe())
    print("\nSummary of test scores (testscr):")
    print(df["testscr"].describe())

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])
    print(f"\nCorrelation between stratio and testscr: {corr:.3f}")

    # Simple linear regression: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()

    print("\nOLS regression of testscr on stratio:")
    print(model.summary())

    # Compare mean test scores across quartiles of stratio
    df["stratio_q"] = pd.qcut(df["stratio"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])
    group_means = df.groupby("stratio_q")["testscr"].mean()
    print("\nMean testscr by student-teacher ratio quartile (lower = fewer students per teacher):")
    print(group_means)


if __name__ == "__main__":
    main()
