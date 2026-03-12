import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables of interest (if any)
    cols = ["testscr", "stratio", "income", "english", "lunch", "calworks", "expenditure", "computer", "students"]
    df = df.dropna(subset=cols)

    # Simple bivariate regression: test score on student-teacher ratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()

    # Multiple regression controlling for key demographics and resources
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks + expenditure + computer + students",
        data=df,
    ).fit()

    corr = df[["testscr", "stratio"]].corr().iloc[0, 1]

    print("Correlation between testscr and stratio:", corr)
    print("\nSimple regression (testscr ~ stratio):")
    print("  coef_stratio:", model_simple.params["stratio"])
    print("  pvalue_stratio:", model_simple.pvalues["stratio"])
    print("  r_squared:", model_simple.rsquared)

    print("\nMultiple regression with controls:")
    print("  coef_stratio:", model_controls.params["stratio"])
    print("  pvalue_stratio:", model_controls.pvalues["stratio"])
    print("  r_squared:", model_controls.rsquared)


if __name__ == "__main__":
    main()

