import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "testscr"])

    correlation, p_value = stats.pearsonr(df["stratio"], df["testscr"])

    model_simple = sm.OLS(df["testscr"], sm.add_constant(df["stratio"])).fit()

    control_variables = ["income", "english", "lunch"]
    available_controls = [column for column in control_variables if column in df.columns]

    model_with_controls = None
    if available_controls:
        predictors = ["stratio"] + available_controls
        design_matrix = sm.add_constant(df[predictors])
        model_with_controls = sm.OLS(df["testscr"], design_matrix).fit()

    print("Number of observations:", len(df))
    print("Mean student-teacher ratio:", df["stratio"].mean())
    print("Mean test score:", df["testscr"].mean())
    print()
    print("Correlation between student-teacher ratio and test scores:")
    print(f"  r = {correlation:.3f}, p-value = {p_value:.3g}")
    print()
    print("OLS regression of testscr on student-teacher ratio (no controls):")
    print(model_simple.summary().as_text())

    if model_with_controls is not None:
        print()
        print("OLS regression with controls (income, english, lunch):")
        print(model_with_controls.summary().as_text())


if __name__ == "__main__":
    main()

