import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables of interest (if any)
    cols = ["testscr", "stratio", "calworks", "lunch", "income", "english"]
    df_model = df[cols].dropna()

    print("Number of observations:", len(df_model))

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    print("\n=== Simple regression: testscr ~ stratio ===")
    print(model_simple.summary())

    # Multiple regression with basic controls
    X_controls = df_model[["stratio", "calworks", "lunch", "income", "english"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()
    print("\n=== Multiple regression with controls ===")
    print(model_controls.summary())

    # Correlation between stratio and testscr
    corr = df_model["stratio"].corr(df_model["testscr"])
    print("\nCorrelation between student-teacher ratio and test scores:", corr)


if __name__ == "__main__":
    main()

