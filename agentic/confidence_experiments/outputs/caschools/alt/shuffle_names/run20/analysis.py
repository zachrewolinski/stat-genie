import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meanings based on info.json
    enrollment = df["english"]  # total enrollment
    teachers = df["students"]  # number of teachers (FTE)
    read_score = df["district"]  # average reading score
    math_score = df["expenditure"]  # average math score

    # Construct key variables
    df["student_teacher_ratio"] = enrollment / teachers
    df["test_score"] = (read_score + math_score) / 2.0

    # Drop any rows with missing values in the variables of interest
    sub = df[["student_teacher_ratio", "test_score"]].dropna()

    # Correlation
    corr = sub["student_teacher_ratio"].corr(sub["test_score"])

    # Simple linear regression: test_score ~ student_teacher_ratio
    X = sm.add_constant(sub["student_teacher_ratio"])
    y = sub["test_score"]
    model = sm.OLS(y, X).fit()

    slope = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    print("N:", len(sub))
    print("Correlation (STR, test_score):", corr)
    print("Slope (effect of +1 in STR):", slope)
    print("p-value (slope):", p_value)
    print("R-squared:", r_squared)

    # Multiple regression with basic controls
    controls = df[["income", "school", "computer", "rownames", "grades"]]
    multi = pd.concat(
        [sub["test_score"], sub["student_teacher_ratio"], controls], axis=1
    ).dropna()

    X_multi = sm.add_constant(
        multi[["student_teacher_ratio", "income", "school", "computer", "rownames", "grades"]]
    )
    y_multi = multi["test_score"]
    model_multi = sm.OLS(y_multi, X_multi).fit()

    slope_multi = model_multi.params["student_teacher_ratio"]
    p_value_multi = model_multi.pvalues["student_teacher_ratio"]

    print("Multiple regression: slope for STR:", slope_multi)
    print("Multiple regression: p-value for STR:", p_value_multi)
    print("Multiple regression: R-squared:", model_multi.rsquared)


if __name__ == "__main__":
    main()
