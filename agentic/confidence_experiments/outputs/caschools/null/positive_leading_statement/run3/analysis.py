import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    df["tsr"] = df["teachers"] / df["students"]  # teachers per student

    # Drop rows with missing values in variables of interest (if any)
    cols_basic = ["testscr", "str", "tsr"]
    cols_controls = cols_basic + [
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "income",
        "english",
    ]
    df_basic = df[cols_basic].dropna()
    df_controls = df[cols_controls].dropna()

    # Simple bivariate regression: testscr ~ str
    X_basic = sm.add_constant(df_basic["str"])
    y_basic = df_basic["testscr"]
    model_basic = sm.OLS(y_basic, X_basic).fit()

    # Multiple regression with controls
    X_controls = sm.add_constant(
        df_controls[
            ["str", "calworks", "lunch", "computer", "expenditure", "income", "english"]
        ]
    )
    y_controls = df_controls["testscr"]
    model_controls = sm.OLS(y_controls, X_controls).fit()

    # Correlation between str and testscr
    corr_str = df_basic["str"].corr(df_basic["testscr"])
    corr_tsr = df_basic["tsr"].corr(df_basic["testscr"])

    # Print summary statistics needed for interpretation
    print("N (basic):", len(df_basic))
    print("N (controls):", len(df_controls))
    print("Mean testscr:", df_basic["testscr"].mean())
    print("Mean str:", df_basic["str"].mean())
    print("Student-teacher ratio summary:")
    print(df_basic["str"].describe())
    print("Correlation (testscr, str):", corr_str)
    print("Correlation (testscr, tsr):", corr_tsr)
    print("\nBivariate regression: testscr ~ str")
    print("Coefficient on str:", model_basic.params["str"])
    print("Std. error (str):", model_basic.bse["str"])
    print("t-stat (str):", model_basic.tvalues["str"])
    print("p-value (str):", model_basic.pvalues["str"])
    print("R-squared:", model_basic.rsquared)

    print("\nMultiple regression: testscr ~ str + controls")
    print("Coefficient on str:", model_controls.params["str"])
    print("Std. error (str):", model_controls.bse["str"])
    print("t-stat (str):", model_controls.tvalues["str"])
    print("p-value (str):", model_controls.pvalues["str"])
    print("R-squared:", model_controls.rsquared)


if __name__ == "__main__":
    main()
