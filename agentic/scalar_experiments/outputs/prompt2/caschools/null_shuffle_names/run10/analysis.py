import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str = "caschools.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # According to info.json, the column names are shuffled relative to their meanings.
    # Mappings based on provided descriptions:
    # - "english"  -> total enrollment (students)
    # - "students" -> number of teachers (FTE)
    # - "district" -> average reading score
    # - "expenditure" -> average math score
    # We derive:
    #   student_teacher_ratio = enrollment / teachers
    #   test_score = average of reading and math scores
    enrollment = df["english"]
    teachers = df["students"]
    read_score = df["district"]
    math_score = df["expenditure"]

    # Guard against division by zero just in case (none expected from metadata).
    ratio = enrollment / teachers.replace(0, np.nan)

    testscr = (read_score + math_score) / 2.0

    # Additional covariates (names describe their semantic meaning per info.json)
    data = pd.DataFrame(
        {
            "testscr": testscr,
            "stratio": ratio,
            "enroll": enrollment,
            "income": df["income"],  # district average income (in USD 1,000)
            "el_pct": df["rownames"],  # percent English learners
            "calw_pct": df["school"],  # percent qualifying for CalWorks
            "lunch_pct": df["computer"],  # percent qualifying for reduced-price lunch
            "expn_stu": df["grades"],  # expenditure per student
        }
    ).dropna()

    return data


def analyze_relationship(data: pd.DataFrame) -> dict:
    """Compute correlation and simple/multiple regressions of test scores on STR."""
    # Correlation between student-teacher ratio and test scores
    corr = data["testscr"].corr(data["stratio"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(data["stratio"])
    model_simple = sm.OLS(data["testscr"], X_simple).fit()

    coef_str_simple = float(model_simple.params["stratio"])
    p_str_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key covariates
    X_multi = sm.add_constant(
        data[["stratio", "income", "el_pct", "calw_pct", "lunch_pct", "expn_stu"]]
    )
    model_multi = sm.OLS(data["testscr"], X_multi).fit()

    coef_str_multi = float(model_multi.params["stratio"])
    p_str_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Group comparison: lowest vs highest STR quartiles
    q1 = data["stratio"].quantile(0.25)
    q3 = data["stratio"].quantile(0.75)
    low_ratio = data[data["stratio"] <= q1]["testscr"]
    high_ratio = data[data["stratio"] >= q3]["testscr"]

    mean_low = float(low_ratio.mean())
    mean_high = float(high_ratio.mean())
    diff_mean = mean_low - mean_high

    return {
        "corr": float(corr),
        "simple": {
            "coef_str": coef_str_simple,
            "p_str": p_str_simple,
            "r2": r2_simple,
        },
        "multiple": {
            "coef_str": coef_str_multi,
            "p_str": p_str_multi,
            "r2": r2_multi,
        },
        "group_diff": {
            "mean_low_str": mean_low,
            "mean_high_str": mean_high,
            "diff_mean": diff_mean,
        },
    }


def main() -> None:
    data = load_data()
    results = analyze_relationship(data)

    # Print results in a structured way so they can be inspected from the outside.
    print(json.dumps(results, indent=2))

    # Optionally, also save raw analysis output for debugging (not required by spec).
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

