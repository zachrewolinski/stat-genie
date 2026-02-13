import pandas as pd
import statsmodels.api as sm


def run_regression(df: pd.DataFrame, label: str) -> None:
    print("=" * 80)
    print(label)
    print("=" * 80)

    corr = df["stratio"].corr(df["avg_score"])
    print("Correlation between stratio and avg_score:", corr)
    print()

    # Simple bivariate regression
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    print("Simple OLS: avg_score ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key covariates
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["avg_score"], X_multi).fit()
    print("Multiple OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())
    print()

    # Group comparison by quartiles of stratio
    df = df.copy()
    df["str_q"] = pd.qcut(df["stratio"], 4, labels=False, duplicates="drop")
    print("Average scores by stratio quartile:")
    print(df.groupby("str_q")["avg_score"].mean())
    print()


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]

    # Overall achievement measure: average of reading and math scores
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    print("Dataset shape:", df.shape)
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print()

    # Full-sample analysis
    run_regression(df, "Full sample")

    # Trim extreme ratios to focus on more plausible range
    trimmed = df[(df["stratio"] >= 10) & (df["stratio"] <= 40)]
    print("Trimmed dataset shape (10 <= stratio <= 40):", trimmed.shape)
    if not trimmed.empty:
        run_regression(trimmed, "Trimmed sample: 10 <= stratio <= 40")


if __name__ == "__main__":
    main()

