import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Map columns to their semantic meaning based on info.json
    enrollment = df["english"]  # Total enrollment
    n_teachers = df["students"]  # Number of teachers

    # Student-teacher ratio (students per teacher)
    stratio = enrollment / n_teachers

    # Academic performance: average of reading and math scores
    read_score = df["district"]  # Average reading score
    math_score = df["expenditure"]  # Average math score
    testscr = (read_score + math_score) / 2.0

    df["stratio"] = stratio
    df["testscr"] = testscr

    # Simple bivariate relationship
    corr = df["stratio"].corr(df["testscr"])

    # Linear regression of test scores on student-teacher ratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()
    slope = model.params["stratio"]
    p_value = model.pvalues["stratio"]

    # Map statistical evidence to Likert-style scalar in [-100, 100]
    # We answer: "Is a lower student-teacher ratio associated with higher academic performance?"
    # A negative slope (higher ratio -> lower scores) supports a "Yes" answer.
    if p_value < 0.001 and slope < 0 and abs(corr) >= 0.3:
        scalar = 80
    elif p_value < 0.01 and slope < 0 and abs(corr) >= 0.2:
        scalar = 60
    elif p_value < 0.05 and slope < 0:
        scalar = 40
    elif p_value >= 0.05:
        scalar = 0
    else:
        # Any significant positive slope would indicate evidence against the claim
        scalar = -40

    # Write scalar conclusion only
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

