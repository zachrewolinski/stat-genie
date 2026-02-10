import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: larger values = more students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Overall test score as the mean of reading and math
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Simple correlation between ratio and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = sm.add_constant(df[["stratio"] + available_controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()

    print("Correlation (testscr vs student-teacher ratio):", corr)
    print("\nSimple regression coefficient on stratio:", model_simple.params["stratio"])
    print("Simple regression p-value for stratio:", model_simple.pvalues["stratio"])

    print("\nControls regression coefficient on stratio:", model_controls.params["stratio"])
    print("Controls regression p-value for stratio:", model_controls.pvalues["stratio"])


if __name__ == "__main__":
    main()

