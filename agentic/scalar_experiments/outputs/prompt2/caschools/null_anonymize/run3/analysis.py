import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    # feature6: total enrollment (students)
    # feature7: number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading (feature14) and math (feature15) scores
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    print("Student-teacher ratio summary:")
    print(df["stratio"].describe(), end="\n\n")

    print("Score summaries (reading, math, average):")
    print(df[["feature14", "feature15", "avg_score"]].describe(), end="\n\n")

    # Correlations between student-teacher ratio and scores
    print("Correlation matrix (stratio vs scores):")
    corr = df[["stratio", "feature14", "feature15", "avg_score"]].corr()
    print(corr, end="\n\n")

    # Correlations in a more typical ratio range to reduce outlier influence
    typical = df[df["stratio"].between(10, 30)]
    if not typical.empty:
        print("Correlation matrix in typical range (10 <= stratio <= 30):")
        corr_typical = typical[["stratio", "feature14", "feature15", "avg_score"]].corr()
        print(corr_typical, end="\n\n")

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    print("OLS: avg_score ~ stratio")
    print(model_simple.summary(), end="\n\n")

    # Multiple regression with key covariates to check robustness
    # feature8: % CalWorks, feature9: % reduced-price lunch,
    # feature11: expenditure per student, feature12: avg income,
    # feature13: % English learners
    covariates = ["stratio", "feature8", "feature9", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["avg_score"], X_multi).fit()
    print("OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()
