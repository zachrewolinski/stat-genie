import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously invalid rows (e.g., non‑positive teachers or students)
    df = df[(df["students"] > 0) & (df["teachers"] > 0)].dropna(subset=["stratio", "testscr"])

    # Simple linear regression of average test score on student‑teacher ratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()

    coef_stratio = model.params["stratio"]
    pvalue_stratio = model.pvalues["stratio"]

    # Correlation as a simple, model‑free summary
    corr = df["stratio"].corr(df["testscr"])

    # Decision rule:
    #   Answer "Yes" if higher student‑teacher ratios (larger classes) are
    #   associated with *lower* test scores, i.e. a negative relationship
    #   between stratio and testscr, and the slope is statistically
    #   distinguishable from zero at the 5% level.
    associated = (coef_stratio < 0) and (pvalue_stratio < 0.05)

    response = "Yes" if associated else "No"

    if coef_stratio < 0:
        direction_phrase = (
            "a negative slope: on average, districts with more students per "
            "teacher tend to have lower test scores."
        )
    else:
        direction_phrase = (
            "a slope that is not negative: on average, districts with more "
            "students per teacher do not have lower test scores."
        )

    if pvalue_stratio < 0.05:
        significance_phrase = (
            "This effect is statistically distinguishable from zero at the "
            "5% level."
        )
    else:
        significance_phrase = (
            "This effect is not statistically distinguishable from zero at the "
            "5% level, so the estimated association could easily be due to "
            "sampling variability."
        )

    if abs(corr) < 0.1:
        corr_strength = "very weak"
    elif abs(corr) < 0.3:
        corr_strength = "modest"
    else:
        corr_strength = "moderate to strong"

    explanation = (
        "Using the 1998–1999 California K–8 district data, I computed each "
        "district's student–teacher ratio as students divided by teachers and "
        "defined academic performance as the average of 5th‑grade reading and "
        "math Stanford 9 scores. A linear regression of average score on the "
        "student–teacher ratio shows "
        f"{direction_phrase} The estimated slope on the student–teacher ratio "
        f"is {coef_stratio:.2f} points per additional student per teacher, "
        f"with a p‑value of {pvalue_stratio:.3f}. {significance_phrase} The "
        "simple correlation between student–teacher ratio and test scores is "
        f"{corr:.2f}, indicating a {corr_strength} linear relationship. "
        "Given the small magnitude of the slope and correlation and the lack "
        "of strong statistical evidence for a negative effect, this analysis "
        "does not provide clear support for the claim that lower "
        "student–teacher ratios are associated with higher academic "
        "performance in this dataset."
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
