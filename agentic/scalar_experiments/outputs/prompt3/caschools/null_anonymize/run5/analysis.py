import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df["stratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Basic correlations between student-teacher ratio and performance
    corr_read = df["stratio"].corr(df["feature14"])
    corr_math = df["stratio"].corr(df["feature15"])
    corr_avg = df["stratio"].corr(df["avg_score"])

    print("Correlation (stratio vs reading):", corr_read)
    print("Correlation (stratio vs math):   ", corr_math)
    print("Correlation (stratio vs average):", corr_avg)

    # Simple linear regressions: scores on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    model_read = sm.OLS(df["feature14"], X_simple).fit()
    model_math = sm.OLS(df["feature15"], X_simple).fit()
    model_avg = sm.OLS(df["avg_score"], X_simple).fit()

    print("\nSimple OLS: Reading on student-teacher ratio")
    print("  beta_stratio:", model_read.params["stratio"])
    print("  p_value:", model_read.pvalues["stratio"])

    print("\nSimple OLS: Math on student-teacher ratio")
    print("  beta_stratio:", model_math.params["stratio"])
    print("  p_value:", model_math.pvalues["stratio"])

    print("\nSimple OLS: Average score on student-teacher ratio")
    print("  beta_stratio:", model_avg.params["stratio"])
    print("  p_value:", model_avg.pvalues["stratio"])

    # Multiple regressions controlling for key socioeconomic covariates
    covariates = ["stratio", "feature8", "feature9", "feature11", "feature12", "feature13"]
    X_cov = sm.add_constant(df[covariates])

    model_read_cov = sm.OLS(df["feature14"], X_cov).fit()
    model_math_cov = sm.OLS(df["feature15"], X_cov).fit()
    model_avg_cov = sm.OLS(df["avg_score"], X_cov).fit()

    print("\nMultiple OLS (with covariates): Reading on stratio + controls")
    print("  beta_stratio:", model_read_cov.params["stratio"])
    print("  p_value:", model_read_cov.pvalues["stratio"])

    print("\nMultiple OLS (with covariates): Math on stratio + controls")
    print("  beta_stratio:", model_math_cov.params["stratio"])
    print("  p_value:", model_math_cov.pvalues["stratio"])

    print("\nMultiple OLS (with covariates): Average score on stratio + controls")
    print("  beta_stratio:", model_avg_cov.params["stratio"])
    print("  p_value:", model_avg_cov.pvalues["stratio"])


if __name__ == "__main__":
    main()

