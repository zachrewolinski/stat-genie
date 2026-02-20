import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: total enrollment divided by number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: reading, math, and their average
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any potential missing values just in case
    df = df[["stratio", "feature14", "feature15", "avg_score"]].dropna()

    # Basic distribution of student-teacher ratio
    print("Number of districts:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print()

    # Correlations
    print("Correlations with student-teacher ratio:")
    print("  Reading vs ratio:", df["stratio"].corr(df["feature14"]))
    print("  Math vs ratio   :", df["stratio"].corr(df["feature15"]))
    print("  Avg vs ratio    :", df["stratio"].corr(df["avg_score"]))
    print()

    # Simple linear regression: avg_score ~ stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["avg_score"], X).fit()

    coef = model.params["stratio"]
    p_value = model.pvalues["stratio"]
    r_squared = model.rsquared

    print("OLS regression: avg_score ~ stratio")
    print("  Coefficient on stratio:", coef)
    print("  p-value for stratio  :", p_value)
    print("  R-squared            :", r_squared)

    # Robustness check: restrict to plausible ratios (e.g., 5 to 40 students per teacher)
    df_trim = df[(df["stratio"] >= 5) & (df["stratio"] <= 40)]
    print()
    print("Trimmed sample (5 <= stratio <= 40):")
    print("  Number of districts:", len(df_trim))
    print("  Avg ratio:", df_trim["stratio"].mean())
    print("  Corr (avg_score vs ratio):", df_trim["stratio"].corr(df_trim["avg_score"]))

    if len(df_trim) > 0:
        X_trim = sm.add_constant(df_trim["stratio"])
        model_trim = sm.OLS(df_trim["avg_score"], X_trim).fit()
        print("  OLS coef on ratio:", model_trim.params["stratio"])
        print("  p-value:", model_trim.pvalues["stratio"])
        print("  R-squared:", model_trim.rsquared)


if __name__ == "__main__":
    main()
