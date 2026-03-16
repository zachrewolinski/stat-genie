import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest
    vars_simple = ["testscr", "stratio"]
    vars_controls = vars_simple + ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    df_simple = df[vars_simple].dropna()
    df_controls = df[vars_controls].dropna()

    print("Number of observations (simple model):", len(df_simple))
    print("Number of observations (with controls):", len(df_controls))

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_simple["stratio"])
    y_simple = df_simple["testscr"]
    model_simple = sm.OLS(y_simple, X_simple).fit()

    coef_stratio_simple = model_simple.params["stratio"]
    p_stratio_simple = model_simple.pvalues["stratio"]

    print("\nSimple OLS: testscr ~ stratio")
    print("Coefficient on stratio:", round(coef_stratio_simple, 3))
    print("P-value for stratio:", p_stratio_simple)

    # Multiple regression with controls
    X_controls = df_controls[["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]]
    X_controls = sm.add_constant(X_controls)
    y_controls = df_controls["testscr"]
    model_controls = sm.OLS(y_controls, X_controls).fit()

    coef_stratio_controls = model_controls.params["stratio"]
    p_stratio_controls = model_controls.pvalues["stratio"]

    print("\nMultiple OLS: testscr ~ stratio + controls")
    print("Coefficient on stratio:", round(coef_stratio_controls, 3))
    print("P-value for stratio:", p_stratio_controls)

    # Also report simple correlation between stratio and testscr
    corr = df[["stratio", "testscr"]].corr().iloc[0, 1]
    print("\nCorrelation between stratio and testscr:", round(corr, 3))


if __name__ == "__main__":
    main()

