import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key columns (if any)
    key_cols = ["stratio", "testscr", "income", "english", "lunch", "calworks", "expenditure"]
    df_clean = df.dropna(subset=key_cols).copy()

    # Simple Pearson correlation between student-teacher ratio and test scores
    corr, p_corr = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    model_simple = sm.OLS(df_clean["testscr"], X_simple).fit()

    # Multiple regression controlling for key covariates
    X_controls = df_clean[["stratio", "income", "english", "lunch", "calworks", "expenditure"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_clean["testscr"], X_controls).fit()

    # Print summary metrics needed for interpretation
    print("Number of observations:", len(df_clean))
    print("\n=== Correlation ===")
    print(f"Pearson correlation (testscr vs stratio): {corr:.4f}")
    print(f"P-value: {p_corr:.4g}")

    print("\n=== Simple OLS: testscr ~ stratio ===")
    print(f"Coefficient on stratio: {model_simple.params['stratio']:.4f}")
    print(f"P-value: {model_simple.pvalues['stratio']:.4g}")
    print(f"R-squared: {model_simple.rsquared:.4f}")

    print("\n=== OLS with controls: testscr ~ stratio + income + english + lunch + calworks + expenditure ===")
    print(f"Coefficient on stratio: {model_controls.params['stratio']:.4f}")
    print(f"P-value: {model_controls.pvalues['stratio']:.4g}")
    print(f"R-squared: {model_controls.rsquared:.4f}")


if __name__ == "__main__":
    main()

