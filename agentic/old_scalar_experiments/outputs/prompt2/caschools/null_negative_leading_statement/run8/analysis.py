import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio (higher = more students per teacher, i.e., larger classes)
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores
    df["score"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "score"])

    # Simple correlations
    corr_score = df["stratio"].corr(df["score"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])

    print(f"N = {len(df)}")
    print(f"Mean stratio = {df['stratio'].mean():.3f}")
    print(f"Mean score = {df['score'].mean():.3f}")
    print("Correlations (stratio with score/read/math):")
    print(
        f"  score: {corr_score:.4f}, read: {corr_read:.4f}, math: {corr_math:.4f}"
    )

    # Compare mean scores across quartiles of the student-teacher ratio
    df["stratio_q"] = pd.qcut(df["stratio"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])
    group_means = df.groupby("stratio_q")["score"].agg(["mean", "count"])
    print("\nMean score by stratio quartile (students per teacher):")
    print(group_means)

    # Simple OLS: score on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["score"]
    model_simple = sm.OLS(y, X_simple).fit()

    print("\nSimple OLS: score ~ stratio")
    print(model_simple.summary())

    # Multiple regression controlling for key demographics/resources
    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()

    print("\nMultiple OLS: score ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()
