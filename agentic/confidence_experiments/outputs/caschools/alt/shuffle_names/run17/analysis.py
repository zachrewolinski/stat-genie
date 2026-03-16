import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Based on info.json, map columns to their semantic meaning:
    # english   -> total enrollment (students)
    # students  -> number of teachers
    # district  -> average reading score
    # expenditure -> average math score
    # school    -> percent on CalWorks (income assistance)
    # computer  -> percent qualifying for reduced-price lunch
    # income    -> district average income (in USD 1,000)
    # rownames  -> percent of English learners

    # Construct student-teacher ratio and overall academic performance
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Basic sanity checks
    print("Number of rows:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])
    print("\nCorrelation between student-teacher ratio and test scores:", corr)

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with common covariates from the CASchools context
    covariates = ["stratio", "income", "school", "computer", "rownames"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()
    print("\nMultiple OLS: testscr ~ stratio + income + school (CalWorks%)"
          " + computer (lunch%) + rownames (English learners%)")
    print(model_multi.summary())

    # Print key coefficients for easier reading
    print("\nKey coefficients (simple model):")
    print(model_simple.params)
    print("p-values:", model_simple.pvalues)

    print("\nKey coefficients (multiple model):")
    print(model_multi.params)
    print("p-values:", model_multi.pvalues)


if __name__ == "__main__":
    main()

