import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Simple bivariate regression: avgscore on stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()

    # Multiple regression including key demographic and resource controls
    controls = ["calworks", "lunch", "income", "english", "computer", "expenditure"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(df["avgscore"], X_controls).fit()

    # Save key results to a text file for inspection
    with open("analysis_results.txt", "w") as f:
        f.write("Simple regression: avgscore ~ stratio\n")
        f.write(model_simple.summary().as_text())
        f.write("\n\n")
        f.write("Controls regression: avgscore ~ stratio + controls\n")
        f.write(model_controls.summary().as_text())


if __name__ == "__main__":
    main()

