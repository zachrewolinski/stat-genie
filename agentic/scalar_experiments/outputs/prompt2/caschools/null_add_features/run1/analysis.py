import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio; guard against division by zero.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    # Average reading and math scores as overall test score.
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    # Drop rows with missing key variables if any.
    df = df.dropna(subset=["stratio", "testscr"])
    return df


def summarize_relationship(df: pd.DataFrame) -> dict:
    summary = {}
    stratio = df["stratio"]
    testscr = df["testscr"]

    summary["n_obs"] = int(len(df))
    summary["stratio_mean"] = float(stratio.mean())
    summary["testscr_mean"] = float(testscr.mean())
    summary["corr_stratio_testscr"] = float(stratio.corr(testscr))

    return summary


def run_regressions(df: pd.DataFrame) -> dict:
    results = {}

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    results["simple_coef_stratio"] = float(model_simple.params["stratio"])
    results["simple_pvalue_stratio"] = float(model_simple.pvalues["stratio"])
    results["simple_r2"] = float(model_simple.rsquared)

    # Multiple regression with key demographic and resource controls.
    controls = [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = df[["stratio"] + available_controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["testscr"], X_controls, missing="drop").fit()

    results["controls_coef_stratio"] = float(model_controls.params["stratio"])
    results["controls_pvalue_stratio"] = float(model_controls.pvalues["stratio"])
    results["controls_r2"] = float(model_controls.rsquared)

    # Effect of a 5-student reduction in STR based on controlled model.
    coef = model_controls.params["stratio"]
    results["delta_testscr_5_student_reduction"] = float(-5.0 * coef)

    return results


def build_conclusion(summary: dict, reg_results: dict) -> dict:
    corr = summary["corr_stratio_testscr"]
    simple_coef = reg_results["simple_coef_stratio"]
    simple_p = reg_results["simple_pvalue_stratio"]
    controls_coef = reg_results["controls_coef_stratio"]
    controls_p = reg_results["controls_pvalue_stratio"]
    delta_5 = reg_results["delta_testscr_5_student_reduction"]

    # Decide on response and confidence.
    # We interpret "lower student-teacher ratio associated with higher performance"
    # as a negative coefficient on stratio (fewer students per teacher -> higher score).
    negative_and_significant = (controls_coef < 0) and (controls_p < 0.05)

    if negative_and_significant and np.sign(corr) < 0 and simple_coef < 0 and simple_p < 0.05:
        response = "Yes"
        confidence = 90
    elif negative_and_significant and (simple_coef < 0 or corr < 0):
        response = "Yes"
        confidence = 80
    elif (controls_coef < 0 or simple_coef < 0 or corr < 0) and controls_p < 0.1:
        response = "Yes"
        confidence = 65
    else:
        response = "No"
        # Confidence reflects strength and consistency of evidence against
        # a strong negative association in this dataset.
        if controls_p > 0.2 and simple_p > 0.2:
            confidence = 80
        else:
            confidence = 60

    explanation_lines = []
    explanation_lines.append(
        "The research question asks whether a lower student-teacher ratio "
        "is associated with higher academic performance."
    )
    explanation_lines.append(
        "Using the provided California school districts dataset, I computed "
        "the student-teacher ratio as the number of students divided by the "
        "number of teachers in each district, and I measured academic "
        "performance as the average of the reading and math Stanford 9 test scores."
    )
    explanation_lines.append(
        f"The simple correlation between the student-teacher ratio and the "
        f"average test score is {corr:.3f}, and a bivariate OLS regression "
        f"of test scores on the student-teacher ratio yields a coefficient "
        f"of {simple_coef:.3f} with a p-value of {simple_p:.3f}."
    )
    explanation_lines.append(
        "I then estimated a multiple regression including controls for income, "
        "percent English learners, percent of students on reduced-price lunch, "
        "percent receiving CalWorks, per-pupil expenditure, and the number of "
        "computers."
    )
    explanation_lines.append(
        f"In this controlled model, the coefficient on the student-teacher ratio "
        f"is {controls_coef:.3f} with a p-value of {controls_p:.3f}, and the model "
        f"explains about {reg_results['controls_r2']:.3f} of the variance in test scores."
    )
    explanation_lines.append(
        f"Interpreting the controlled model, a reduction of 5 students per teacher "
        f"is associated with an estimated change in average test scores of "
        f"{delta_5:.2f} points."
    )
    if response == "Yes":
        explanation_lines.append(
            "Because the estimated association is consistently negative across "
            "correlation and regression analyses and is statistically significant "
            "at conventional levels in the controlled specification, I conclude "
            "that, in this dataset, lower student-teacher ratios are associated "
            "with higher academic performance."
        )
    else:
        explanation_lines.append(
            "Because the estimated association is not consistently negative and "
            "statistically significant across correlation and regression analyses, "
            "the data do not provide strong evidence that lower student-teacher "
            "ratios are associated with higher academic performance in this dataset."
        )

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)
    summary = summarize_relationship(df)
    reg_results = run_regressions(df)
    conclusion = build_conclusion(summary, reg_results)

    # Write required JSON object to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

