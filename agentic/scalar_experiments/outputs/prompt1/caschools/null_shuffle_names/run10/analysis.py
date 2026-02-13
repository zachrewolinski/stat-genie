import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # According to info.json, the semantic mappings are:
    # - "english": total enrollment (number of students)
    # - "students": number of teachers
    # - "district": average reading score
    # - "expenditure": average math score
    #
    # Define student-teacher ratio and academic performance.
    students_total = pd.to_numeric(df["english"], errors="coerce")
    teachers = pd.to_numeric(df["students"], errors="coerce")
    read_score = pd.to_numeric(df["district"], errors="coerce")
    math_score = pd.to_numeric(df["expenditure"], errors="coerce")

    # Drop rows with missing or invalid values and compute derived variables.
    valid_mask = (
        students_total.notna()
        & teachers.notna()
        & (teachers > 0)
        & read_score.notna()
        & math_score.notna()
    )
    data = pd.DataFrame(
        {
            "students_total": students_total[valid_mask],
            "teachers": teachers[valid_mask],
            "read_score": read_score[valid_mask],
            "math_score": math_score[valid_mask],
        }
    )
    data["stratio"] = data["students_total"] / data["teachers"]
    data["testscr"] = (data["read_score"] + data["math_score"]) / 2.0

    # Simple linear regression: test score on student-teacher ratio.
    X = sm.add_constant(data["stratio"])
    model = sm.OLS(data["testscr"], X).fit()

    slope = float(model.params["stratio"])
    p_value = float(model.pvalues["stratio"])
    r_squared = float(model.rsquared)
    corr = float(data["stratio"].corr(data["testscr"]))

    # Decide on Yes/No based on sign and significance of the slope.
    if slope < 0 and p_value < 0.05:
        response = "Yes"
        association_text = (
            "Districts with lower student-teacher ratios tend to have higher test scores."
        )
    else:
        response = "No"
        if slope < 0:
            association_text = (
                "The estimated relationship is negative but not statistically significant."
            )
        else:
            association_text = (
                "The estimated relationship is not negative, so the data do not support "
                "higher scores at lower student-teacher ratios."
            )

    if response == "Yes":
        evidence_text = (
            "Because the slope is negative and statistically different from zero at "
            "conventional levels, this provides evidence that lower student-teacher "
            "ratios are associated with higher academic performance."
        )
    else:
        evidence_text = (
            "Because the slope is not both negative and statistically significant at "
            "the 5% level, the data do not provide strong evidence that lower "
            "student-teacher ratios are associated with higher academic performance."
        )

    explanation = (
        "Using data from 420 California K-6 and K-8 school districts, I computed the "
        "student-teacher ratio as total enrollment divided by the number of teachers and "
        "measured academic performance as the average of district reading and math scores. "
        f"A linear regression of test scores on the student-teacher ratio yields an estimated "
        f"slope of {slope:.3f} with a p-value of {p_value:.3g} and an R-squared of {r_squared:.3f}. "
        f"The correlation between the student-teacher ratio and test scores is {corr:.3f}. "
        f"{association_text} {evidence_text}"
    )

    result = {"response": response, "explanation": explanation}

    # Write the required JSON output file with no extra text.
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
