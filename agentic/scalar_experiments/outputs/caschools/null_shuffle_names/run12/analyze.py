import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to info.json descriptions:
    # - "english" corresponds to total enrollment (students).
    # - "students" corresponds to number of teachers.
    # - "district" is average reading score.
    # - "expenditure" is average math score.
    #
    # We define the student-teacher ratio as students per teacher.
    student_teacher_ratio = df["english"] / df["students"]

    # Academic performance: average of reading and math scores.
    test_score = (df["district"] + df["expenditure"]) / 2.0

    analysis_df = pd.DataFrame(
        {
            "str": student_teacher_ratio,
            "testscr": test_score,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

    # Basic correlation.
    corr = analysis_df["str"].corr(analysis_df["testscr"])

    # Simple linear regression: testscr ~ str.
    X = sm.add_constant(analysis_df["str"])
    model = sm.OLS(analysis_df["testscr"], X).fit()
    slope = float(model.params["str"])
    t_value = float(model.tvalues["str"])

    # Map results to Likert-scale scalar in [-100, 100].
    # Lower STR associated with higher performance corresponds to:
    #   negative slope and negative correlation.
    # We scale by combined signal strength (|corr| and |t|),
    # then clamp to [-100, 100].
    signal = -slope  # positive if lower STR -> higher score

    if math.isnan(signal) or math.isnan(corr) or math.isnan(t_value):
        scalar = 0
    else:
        direction = 1.0 if (signal > 0 and corr < 0) else -1.0 if (signal < 0 and corr > 0) else 0.0
        strength = min(1.0, (abs(corr) + min(abs(t_value) / 10.0, 1.0)) / 2.0)
        scalar_float = direction * 100.0 * strength
        scalar = int(round(scalar_float))

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

