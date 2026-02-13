import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Summary of key variables:")
    print(df[["stratio", "testscr", "students", "teachers"]].describe())
    print()

    # Simple correlations
    corr_testscr = df["stratio"].corr(df["testscr"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    print(f"Correlation between student-teacher ratio and test score: {corr_testscr:.4f}")
    print(f"Correlation between student-teacher ratio and reading score: {corr_read:.4f}")
    print(f"Correlation between student-teacher ratio and math score: {corr_math:.4f}")
    print()

    # Bivariate OLS: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    print("Bivariate regression: testscr ~ stratio")
    print(model1.summary())
    print()

    # Multivariate OLS controlling for key demographics and resources
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    X2 = sm.add_constant(df[covariates])
    model2 = sm.OLS(df["testscr"], X2).fit()
    print("Multivariate regression: testscr ~ stratio + controls")
    print(model2.summary())
    print()

    # Difference in mean scores across quartiles of stratio
    df["stratio_q"] = pd.qcut(df["stratio"], 4, labels=False)
    print("Mean test scores by student-teacher ratio quartile (0=smallest classes, 3=largest):")
    print(df.groupby("stratio_q")["testscr"].mean())
    print()

    # Robustness check: restrict to reasonably sized classes
    trimmed = df[(df["stratio"] >= 5) & (df["stratio"] <= 35)].copy()
    print(f"Trimmed sample size (5 <= stratio <= 35): {len(trimmed)}")
    X_trim = sm.add_constant(trimmed["stratio"])
    model_trim = sm.OLS(trimmed["testscr"], X_trim).fit()
    print("Trimmed bivariate regression: testscr ~ stratio (5 <= stratio <= 35)")
    print(model_trim.summary())
    print()


if __name__ == "__main__":
    main()
