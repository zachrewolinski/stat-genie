import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning based on info.json
    enrollment = df["english"].astype(float)  # total enrollment (students)
    n_teachers = df["students"].astype(float)  # number of teachers (FTE)

    # Compute student-teacher ratio
    df["stratio"] = enrollment / n_teachers

    # Academic performance: average of reading and math scores
    read_score = df["district"].astype(float)  # average reading score
    math_score = df["expenditure"].astype(float)  # average math score
    df["testscr"] = (read_score + math_score) / 2.0

    # Descriptive stats for key variables
    stratio = df["stratio"]
    testscr = df["testscr"]

    print("Student-teacher ratio: mean", stratio.mean(), "min", stratio.min(), "max", stratio.max())
    print("Test score (avg of reading & math): mean", testscr.mean(), "min", testscr.min(), "max", testscr.max())

    # Simple correlation
    corr = stratio.corr(testscr)

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(stratio)
    model_simple = sm.OLS(testscr, X_simple).fit()

    # Multiple OLS with key socioeconomic controls
    controls = ["income", "school", "computer", "rownames", "grades"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(testscr, X_multi).fit()

    print("Correlation (stratio, testscr):", corr)
    print(
        "Simple OLS - beta_stratio:",
        model_simple.params["stratio"],
        "p-value:",
        model_simple.pvalues["stratio"],
        "R-squared:",
        model_simple.rsquared,
    )
    print(
        "Multiple OLS - beta_stratio:",
        model_multi.params["stratio"],
        "p-value:",
        model_multi.pvalues["stratio"],
        "R-squared:",
        model_multi.rsquared,
    )


if __name__ == "__main__":
    main()
