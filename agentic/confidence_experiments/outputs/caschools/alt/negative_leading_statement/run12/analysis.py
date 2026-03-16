import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlations
    corr_str_testscr = df["stratio"].corr(df["testscr"])
    corr_str_read = df["stratio"].corr(df["read"])
    corr_str_math = df["stratio"].corr(df["math"])

    print("Correlation(stratio, testscr):", corr_str_testscr)
    print("Correlation(stratio, read):   ", corr_str_read)
    print("Correlation(stratio, math):   ", corr_str_math)

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    mod_simple = sm.OLS(df["testscr"], X_simple, missing="drop").fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(mod_simple.summary())

    # Multiple regression with key demographic controls
    controls = ["income", "english", "lunch", "calworks"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    mod_controls = sm.OLS(df["testscr"], X_controls, missing="drop").fit()
    print("\nMultiple OLS: testscr ~ stratio + controls")
    print(mod_controls.summary())


if __name__ == "__main__":
    main()

