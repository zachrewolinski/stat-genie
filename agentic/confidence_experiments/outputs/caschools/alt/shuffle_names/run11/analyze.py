import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(info_path: Path):
    with info_path.open() as f:
        info = json.load(f)
    fields = {f["column"]: f["properties"] for f in info["data_desc"]["fields"]}
    return info, fields


def identify_variables(fields):
    """
    Identify the student-teacher ratio variable and test score variables
    based on the textual descriptions in info.json.

    If a direct ratio variable is not documented, we will instead
    identify enrollment and teacher-count variables so that the
    ratio can be constructed as enrollment / teachers.
    """
    str_var = None
    enroll_var = None
    teachers_var = None
    read_var = None
    math_var = None

    for name, props in fields.items():
        desc = (props.get("description") or "").lower()
        if (
            "student-teacher ratio" in desc
            or "student/teacher ratio" in desc
            or "students per teacher" in desc
        ):
            str_var = name
        if "total enrollment" in desc or "total enrolment" in desc:
            enroll_var = name
        if "number of teachers" in desc or "teachers (measured as" in desc:
            teachers_var = name
        if "average reading score" in desc:
            read_var = name
        if "average math score" in desc:
            math_var = name

    return str_var, enroll_var, teachers_var, read_var, math_var


def run_analysis():
    base = Path(__file__).parent
    info, fields = load_metadata(base / "info.json")

    str_var, enroll_var, teachers_var, read_var, math_var = identify_variables(fields)

    if str_var is None and (enroll_var is None or teachers_var is None):
        raise RuntimeError(
            "Could not identify a student-teacher ratio variable or the underlying "
            "enrollment and teacher-count variables from metadata."
        )
    if read_var is None or math_var is None:
        raise RuntimeError("Could not identify reading/math score variables from metadata.")

    df = pd.read_csv(base / "caschools.csv")

    # Create overall test score as the mean of reading and math scores
    df["testscr"] = df[[read_var, math_var]].mean(axis=1)

    # Drop rows with missing values in key variables
    if str_var is not None:
        df["str_ratio"] = df[str_var]
    else:
        df["str_ratio"] = df[enroll_var] / df[teachers_var]

    data = df[["str_ratio", "testscr"]].dropna()

    # Regress testscr on student-teacher ratio (simple OLS)
    X = sm.add_constant(data["str_ratio"])
    y = data["testscr"]
    model = sm.OLS(y, X).fit()

    coef = model.params["str_ratio"]
    p_value = model.pvalues["str_ratio"]
    r_squared = model.rsquared

    # Determine Likert-scale response based on direction, magnitude, and significance
    alpha = 0.05
    if p_value >= alpha:
        # No statistically significant relationship
        response = 30
        explanation = (
            "There is insufficient statistical evidence at the 5% level that the "
            "student-teacher ratio is associated with academic performance. "
            f"The estimated effect of a one-unit increase in the student-teacher ratio "
            f"on the average test score is {coef:.3f}, with p-value {p_value:.3f} and "
            f"R-squared {r_squared:.3f}, indicating a weak and statistically "
            "non-significant relationship."
        )
    else:
        # Statistically significant relationship: map strength and direction to scale
        # Negative coefficient means lower ratio (smaller classes) -> higher scores.
        # Use standardized effect size for scaling.
        std_ratio = data["str_ratio"].std()
        std_testscr = data["testscr"].std()
        standardized_effect = coef * std_ratio / std_testscr

        # Cap standardized effect for scaling
        capped_effect = max(min(standardized_effect, 0.5), -0.5)

        # Base confidence from significance and fit
        if p_value < 0.001:
            base_score = 90
        elif p_value < 0.01:
            base_score = 80
        else:
            base_score = 70

        # Adjust based on strength of association and sign
        strength_adjust = int((abs(capped_effect) / 0.5) * 10)

        if coef < 0:
            # Lower ratio associated with higher performance -> "Yes"
            response = min(100, base_score + strength_adjust)
            direction_text = "lower student-teacher ratios are associated with higher academic performance"
        else:
            # Higher ratio associated with higher performance -> "Yes", but opposite of expectation
            response = min(100, base_score + strength_adjust)
            direction_text = "higher student-teacher ratios are associated with higher academic performance"

        explanation = (
            f"There is strong statistical evidence at the 5% level that the student-teacher ratio "
            f"is associated with academic performance. The estimated coefficient from a linear "
            f"regression of average test scores on the student-teacher ratio is {coef:.3f} "
            f"(p-value {p_value:.4g}, R-squared {r_squared:.3f}), implying that {direction_text}. "
            f"A one-standard-deviation change in the student-teacher ratio corresponds to an "
            f"estimated {standardized_effect:.2f}-standard-deviation change in test scores. "
            "Taken together, this provides a clear 'Yes' answer that a relationship exists, "
            "with the response strength reflecting both the statistical significance and the "
            "moderate practical magnitude of the association."
        )

    conclusion = {"response": int(response), "explanation": explanation}

    with (base / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    run_analysis()
