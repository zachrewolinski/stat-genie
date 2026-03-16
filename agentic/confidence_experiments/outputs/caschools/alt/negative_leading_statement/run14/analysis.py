import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic summary of key variables
    print("Number of districts:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Simple bivariate association
    corr = df["testscr"].corr(df["stratio"])
    print("\nCorrelation between testscr and stratio:", corr)

    simple_model = smf.ols("testscr ~ stratio", data=df).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(simple_model.summary())

    # Multiple regression controlling for observable covariates
    mult_model = smf.ols(
        "testscr ~ stratio + expenditure + income + english + lunch + calworks",
        data=df,
    ).fit()
    print("\nMultiple OLS with controls:")
    print(mult_model.summary())

    # Highlight coefficient and p-value for student-teacher ratio
    coef_str = mult_model.params["stratio"]
    pval_str = mult_model.pvalues["stratio"]
    print("\nKey coefficient (stratio) from multiple regression:")
    print("coef =", coef_str, "p-value =", pval_str)


if __name__ == "__main__":
    main()

