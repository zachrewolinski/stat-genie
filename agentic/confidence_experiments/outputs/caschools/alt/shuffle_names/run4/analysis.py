import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions:
    # - "english" is total enrollment (students)
    # - "students" is number of teachers
    # Student–teacher ratio = students per teacher.
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: use the mean of average reading and math scores.
    # - "district": average reading score
    # - "expenditure": average math score
    df["testscr_mean"] = (df["district"] + df["expenditure"]) / 2.0

    print("Basic summary:")
    print(df[["stratio", "testscr_mean"]].describe())
    corr = df["stratio"].corr(df["testscr_mean"])
    print("\nCorrelation between student–teacher ratio and test score mean:", corr)

    # Simple bivariate regression: testscr_mean ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["testscr_mean"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nBivariate OLS: testscr_mean ~ stratio")
    print(model_simple.summary())

    # Multiple regression with key demographic and resource controls:
    # - "income": district average income
    # - "grades": expenditure per student
    # - "school": percent CalWorks
    # - "computer": percent reduced-price lunch
    # - "rownames": percent English learners
    controls = df[["income", "grades", "school", "computer", "rownames"]]
    X_multi = sm.add_constant(pd.concat([df["stratio"], controls], axis=1))
    model_multi = sm.OLS(y, X_multi).fit()
    print("\nMultivariate OLS with controls:")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

