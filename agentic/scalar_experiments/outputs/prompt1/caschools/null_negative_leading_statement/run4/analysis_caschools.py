import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    vars_used = ["testscr", "stratio", "income", "english", "lunch", "calworks"]
    df_model = df[vars_used].dropna()

    # Simple association: correlation
    corr = df_model["stratio"].corr(df_model["testscr"])
    print(f"Correlation between student-teacher ratio and test score: {corr:.3f}")

    # Simple linear regression
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(f"Coefficient on stratio: {model_simple.params['stratio']:.3f}")
    print(f"p-value for stratio: {model_simple.pvalues['stratio']:.4g}")
    print(f"R-squared: {model_simple.rsquared:.3f}")

    # Multiple regression with key controls
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks", data=df_model
    ).fit()
    print("\nOLS with controls: testscr ~ stratio + income + english + lunch + calworks")
    print(f"Coefficient on stratio: {model_controls.params['stratio']:.3f}")
    print(f"p-value for stratio: {model_controls.pvalues['stratio']:.4g}")
    print(f"R-squared: {model_controls.rsquared:.3f}")


if __name__ == "__main__":
    main()

