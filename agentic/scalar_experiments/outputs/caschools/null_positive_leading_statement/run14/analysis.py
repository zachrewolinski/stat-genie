import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Basic descriptive statistics
    print("Number of districts:", len(df))
    print()
    print("Student-teacher ratio (stratio) summary:")
    print(df["stratio"].describe())
    print()
    print("Average test score (avg_score) summary:")
    print(df["avg_score"].describe())
    print()

    # Simple correlation between student-teacher ratio and test scores
    corr = df["stratio"].corr(df["avg_score"])
    print(f"Correlation between stratio and avg_score: {corr:.4f}")
    print()

    # Simple bivariate regression: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("Bivariate OLS: avg_score ~ stratio")
    print(model_simple.summary())
    print()

    # Multivariate regression with common controls
    controls = ["calworks", "lunch", "english", "income"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()
    print("Multivariate OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

