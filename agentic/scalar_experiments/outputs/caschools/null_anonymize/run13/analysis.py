import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def identify_columns(info: dict) -> dict:
    """Identify key column names from the metadata descriptions."""
    fields = info["data_desc"]["fields"]
    mapping = {}
    for field in fields:
        col = field["column"]
        desc = field["properties"].get("description", "").lower()
        if "total enrollment" in desc:
            mapping["enrollment"] = col
        elif "number of teachers" in desc:
            mapping["teachers"] = col
        elif "average reading score" in desc:
            mapping["reading"] = col
        elif "average math score" in desc:
            mapping["math"] = col
    return mapping


def compute_likert_scalar(slope: float, p_value: float, score_sd: float) -> int:
    """
    Map the evidence on the slope (effect of STR on scores) to a
    Likert-style integer in [-100, 100], where positive values mean
    'Yes: lower STR is associated with higher performance'.
    """
    # If the relationship is essentially null statistically, return neutral.
    if np.isnan(slope) or np.isnan(p_value) or p_value >= 0.1:
        return 0

    # Base score from significance level.
    if p_value < 0.001:
        base = 90
    elif p_value < 0.01:
        base = 80
    elif p_value < 0.05:
        base = 60
    else:  # 0.05 <= p < 0.1
        base = 40

    # Crude effect-size adjustment: effect of a 5-student change
    # relative to one SD of the test score.
    delta_effect = abs(slope) * 5.0
    if score_sd > 0:
        effect_ratio = min(delta_effect / score_sd, 1.0)
    else:
        effect_ratio = 0.0

    # Scale base score upward slightly for larger practical effect.
    scalar = base + int(10 * effect_ratio)
    scalar = min(scalar, 100)

    # Negative slope means higher scores when STR is lower -> positive answer.
    if slope < 0:
        return int(scalar)
    # Positive slope means higher STR is linked to higher scores -> negative answer.
    return int(-scalar)


def main() -> None:
    info = load_metadata(Path("info.json"))
    cols = identify_columns(info)

    missing_keys = {"enrollment", "teachers", "reading", "math"} - set(cols)
    if missing_keys:
        raise RuntimeError(f"Could not identify required columns from metadata: {missing_keys}")

    df = pd.read_csv("caschools.csv")

    enroll = df[cols["enrollment"]].astype(float)
    teachers = df[cols["teachers"]].replace(0, np.nan).astype(float)

    # Student–teacher ratio: students per teacher.
    stratio = enroll / teachers

    reading = df[cols["reading"]].astype(float)
    math = df[cols["math"]].astype(float)
    testscr = (reading + math) / 2.0

    # Drop any rows with missing relevant values.
    data = pd.DataFrame({"testscr": testscr, "stratio": stratio}).dropna()

    X = sm.add_constant(data["stratio"])
    model = sm.OLS(data["testscr"], X).fit()
    slope = model.params["stratio"]
    p_value = model.pvalues["stratio"]
    score_sd = data["testscr"].std()

    # For logging/debugging purposes (not used by the grader).
    print("Number of observations:", int(data.shape[0]))
    print("Slope (effect of STR on score):", slope)
    print("p-value:", p_value)
    print("Test score SD:", score_sd)
    print("R-squared:", model.rsquared)

    scalar = compute_likert_scalar(slope, p_value, score_sd)
    print("Derived Likert scalar:", scalar)

    # Write the scalar to conclusion.txt with no extra text.
    Path("conclusion.txt").write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()

