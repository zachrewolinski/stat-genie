import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map columns to their semantic meanings based on info.json descriptions.
    enroll = df["english"]  # total enrollment
    teachers = df["students"]  # number of teachers

    # Guard against division-by-zero, though it should not occur in this dataset.
    student_teacher_ratio = enroll / teachers.replace(0, np.nan)
    student_teacher_ratio.name = "str"

    read_score = df["district"]  # average reading score
    math_score = df["expenditure"]  # average math score

    # Use overall academic performance as the average of reading and math scores.
    testscr = (read_score + math_score) / 2.0

    # Basic descriptive statistics and correlation
    corr = testscr.corr(student_teacher_ratio)

    # Simple linear regression: academic performance on student-teacher ratio
    simple_X = sm.add_constant(student_teacher_ratio)
    simple_model = sm.OLS(testscr, simple_X, missing="drop").fit()

    # Multiple regression controlling for key observed covariates:
    # income, percent in CalWorks, percent on reduced-price lunch, percent English learners,
    # and expenditure per student.
    income = df["income"]
    calworks_pct = df["school"]
    lunch_pct = df["computer"]
    english_learner_pct = df["rownames"]
    exp_per_student = df["grades"]

    controls = pd.DataFrame(
        {
            "str": student_teacher_ratio,
            "income": income,
            "calworks_pct": calworks_pct,
            "lunch_pct": lunch_pct,
            "english_learner_pct": english_learner_pct,
            "exp_per_student": exp_per_student,
        }
    )

    multi_X = sm.add_constant(controls)
    multi_model = sm.OLS(testscr, multi_X, missing="drop").fit()

    results = {
        "n_obs": int(testscr.dropna().shape[0]),
        "corr_testscr_str": float(corr),
        "simple_coef_str": float(simple_model.params["str"]),
        "simple_pvalue_str": float(simple_model.pvalues["str"]),
        "simple_r2": float(simple_model.rsquared),
        "multi_coef_str": float(multi_model.params["str"]),
        "multi_pvalue_str": float(multi_model.pvalues["str"]),
        "multi_r2": float(multi_model.rsquared),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
