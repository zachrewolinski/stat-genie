import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    model_vars_simple = ["testscr", "stratio"]
    model_vars_controls = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]

    df_simple = df[model_vars_simple].dropna()
    df_controls = df[model_vars_controls].dropna()

    # Correlation between student-teacher ratio and test scores
    corr = df_simple["testscr"].corr(df_simple["stratio"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_simple["stratio"])
    y_simple = df_simple["testscr"]
    model_simple = sm.OLS(y_simple, X_simple).fit()

    # Multiple regression with key demographic and resource controls
    X_controls = df_controls[
        ["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    ]
    X_controls = sm.add_constant(X_controls)
    y_controls = df_controls["testscr"]
    model_controls = sm.OLS(y_controls, X_controls).fit()

    print("Number of districts (simple model):", len(df_simple))
    print("Number of districts (controls model):", len(df_controls))
    print()
    print("Correlation between student-teacher ratio and test scores:")
    print(f"  corr(testscr, stratio) = {corr:.3f}")
    print()
    print("Simple regression: testscr ~ stratio")
    print(model_simple.summary())
    print()
    print("Regression with controls: testscr ~ stratio + income + english + lunch + calworks + computer + expenditure")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

