import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Based on the metadata in info.json:
    # feature6: total enrollment, feature7: number of teachers
    # feature14: average reading score, feature15: average math score
    df["stratio"] = df["feature6"] / df["feature7"]
    df["test_score_avg"] = df[["feature14", "feature15"]].mean(axis=1)

    print("Basic description of student-teacher ratio (students per teacher):")
    print(df["stratio"].describe())
    print()

    print("Correlation between student-teacher ratio and performance:")
    for col in ["feature14", "feature15", "test_score_avg"]:
        corr = df["stratio"].corr(df[col])
        print(f"  stratio vs {col}: corr = {corr:.4f}")
    print()

    # Trim extreme ratios (1st and 99th percentiles) to check robustness
    lower, upper = df["stratio"].quantile([0.01, 0.99])
    df_trim = df[(df["stratio"] >= lower) & (df["stratio"] <= upper)].copy()
    print("After trimming extreme 1% tails of stratio:")
    print(df_trim["stratio"].describe())
    print("Correlation (trimmed) between student-teacher ratio and performance:")
    for col in ["feature14", "feature15", "test_score_avg"]:
        corr = df_trim["stratio"].corr(df_trim[col])
        print(f"  stratio vs {col}: corr = {corr:.4f}")
    print()

    # Simple bivariate regression: test_score_avg ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["test_score_avg"]
    model_simple = sm.OLS(y, X).fit()
    print("Bivariate OLS: test_score_avg ~ stratio")
    print(model_simple.summary())
    print()

    # Multivariate regression controlling for key demographics and resources:
    # feature8: % CalWorks, feature9: % reduced-price lunch, feature11: expenditure per student,
    # feature12: district average income, feature13: % English learners
    covariates = ["stratio", "feature8", "feature9", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(y, X_multi).fit()
    print("Multivariate OLS: test_score_avg ~ stratio + demographics/resources")
    print(model_multi.summary())


if __name__ == "__main__":
    main()
