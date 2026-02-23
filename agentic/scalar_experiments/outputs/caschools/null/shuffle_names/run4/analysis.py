import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and overall test score
    # According to metadata: "english" is total enrollment, "students" is number of teachers.
    df["stratio"] = df["english"] / df["students"]
    # "district" and "expenditure" are average reading and math scores.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    cols = ["testscr", "stratio", "income", "school", "computer", "rownames", "grades"]
    sub = df[cols].dropna()

    print(f"N={len(sub)}")
    print("Summary of student-teacher ratio (stratio):")
    print(sub["stratio"].describe())
    print("Summary of overall test scores (testscr):")
    print(sub["testscr"].describe())

    corr = sub["testscr"].corr(sub["stratio"])
    print(f"Correlation between testscr and stratio: {corr:.4f}")

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(sub[["stratio"]])
    model1 = sm.OLS(sub["testscr"], X1).fit()
    print("\nSimple regression results: testscr ~ stratio")
    print(model1.summary())

    # Multiple regression with standard covariates from metadata
    covariates = ["stratio", "income", "school", "computer", "rownames", "grades"]
    X2 = sm.add_constant(sub[covariates])
    model2 = sm.OLS(sub["testscr"], X2).fit()
    print("\nMultiple regression results: testscr ~ stratio + controls")
    print(model2.summary())

    # Key statistics for downstream reasoning
    print("\nKEY_STATS")
    print(f"coef_simple={model1.params['stratio']}")
    print(f"pval_simple={model1.pvalues['stratio']}")
    print(f"coef_multiple={model2.params['stratio']}")
    print(f"pval_multiple={model2.pvalues['stratio']}")
    print(f"r2_simple={model1.rsquared}")
    print(f"r2_multiple={model2.rsquared}")


if __name__ == "__main__":
    main()

