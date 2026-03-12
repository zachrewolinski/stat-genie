import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def map_to_likert(sign: float, p_value: float, corr_abs: float) -> int:
    """
    Map evidence about the association to a 0–100 Likert-scale integer.

    0   = strong "No" (evidence against the hypothesized direction)
    100 = strong "Yes" (strong evidence in the hypothesized direction)
    """
    # Guard against NaNs
    if np.isnan(sign) or np.isnan(p_value) or np.isnan(corr_abs):
        return 50

    # Hypothesized direction is negative (lower ratio -> higher performance)
    if sign < 0:
        if p_value < 0.001 and corr_abs >= 0.3:
            return 90
        if p_value < 0.01 and corr_abs >= 0.2:
            return 82
        if p_value < 0.05 and corr_abs >= 0.1:
            return 72
        if p_value < 0.1:
            return 60
        return 55

    # Evidence runs against the hypothesized direction
    if p_value < 0.001 and corr_abs >= 0.3:
        return 10
    if p_value < 0.01 and corr_abs >= 0.2:
        return 20
    if p_value < 0.05 and corr_abs >= 0.1:
        return 30
    if p_value < 0.1:
        return 40
    return 45


def format_p_value(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def main() -> None:
    # Load metadata and data
    info_path = Path("info.json")
    data_path = Path("caschools.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Construct key variables based on metadata descriptions:
    # - Student–teacher ratio = total enrollment / number of teachers
    #   (feature6 / feature7)
    # - Academic performance = average of reading and math scores
    #   (feature14 and feature15)
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Variables used as basic socio-economic/demographic controls
    control_cols = ["feature8", "feature9", "feature11", "feature12", "feature13"]

    cols_for_analysis = ["student_teacher_ratio", "avg_test_score"] + control_cols
    df_analysis = df[cols_for_analysis].replace([np.inf, -np.inf], np.nan).dropna()

    # Simple correlation between student–teacher ratio and average test score
    r, p_corr = stats.pearsonr(
        df_analysis["student_teacher_ratio"], df_analysis["avg_test_score"]
    )

    # Simple linear regression: avg_test_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df_analysis["student_teacher_ratio"])
    model_simple = sm.OLS(df_analysis["avg_test_score"], X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    p_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key covariates
    X_multi = df_analysis[["student_teacher_ratio"] + control_cols]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_analysis["avg_test_score"], X_multi).fit()
    coef_multi = float(model_multi.params["student_teacher_ratio"])
    p_multi = float(model_multi.pvalues["student_teacher_ratio"])
    r2_multi = float(model_multi.rsquared)

    # Evidence direction is based primarily on the multiple regression
    sign = np.sign(coef_multi if not np.isnan(coef_multi) else coef_simple)
    response = map_to_likert(sign, p_multi, abs(r))

    dataset_description = info.get("data_desc", {}).get("dataset_description", "")

    explanation_lines = []
    explanation_lines.append(
        "I used the provided California school district dataset to assess whether "
        "lower student–teacher ratios are associated with higher academic performance."
    )
    if dataset_description:
        explanation_lines.append(dataset_description)
    explanation_lines.append(
        "I constructed a student–teacher ratio as total enrollment divided by the "
        "number of teachers and defined academic performance as the average of the "
        "district reading and math scores."
    )
    explanation_lines.append(
        "First, I examined the simple Pearson correlation between the student–teacher "
        f"ratio and the average test score. The correlation was r = {r:.3f} with "
        f"p {format_p_value(p_corr)}, indicating that districts with more students "
        "per teacher tend to have lower test scores when the correlation is negative."
    )
    explanation_lines.append(
        "I then ran a simple linear regression of average test score on the "
        f"student–teacher ratio. The estimated slope was {coef_simple:.3f} "
        f"(p {format_p_value(p_simple)}, R² = {r2_simple:.3f}), so each additional "
        "student per teacher is associated with this many points change in the "
        "average test score."
    )
    explanation_lines.append(
        "To account for observable socio-economic and demographic differences across "
        "districts, I estimated a multiple regression that included the student–teacher "
        "ratio along with controls for the percentage of students in income assistance "
        "programs, the percentage eligible for reduced-price lunch, expenditures per "
        "student, average district income, and the percentage of English learners."
    )
    explanation_lines.append(
        "In this multiple regression, the coefficient on the student–teacher ratio "
        f"was {coef_multi:.3f} with p {format_p_value(p_multi)} and overall model "
        f"R² = {r2_multi:.3f}."
    )

    if sign < 0:
        if p_simple < 0.05 and p_multi < 0.05:
            interpretation = (
                "Both the simple and multiple regression models show a negative and "
                "statistically significant association between the student–teacher "
                "ratio and academic performance, providing strong evidence that "
                "districts with lower student–teacher ratios tend to have higher test "
                "scores, even after adjusting for key socio-economic and "
                "demographic factors."
            )
        elif p_simple < 0.05 and p_multi >= 0.05:
            interpretation = (
                "The simple correlation and regression reveal a negative and "
                "statistically significant association, but after adjusting for "
                "socio-economic and demographic controls the coefficient on the "
                "student–teacher ratio becomes smaller and is no longer "
                "statistically significant. This pattern suggests that districts "
                "with lower student–teacher ratios do tend to have higher academic "
                "performance, but much of this association is explained by other "
                "observed factors, so the independent evidence for a strong effect of "
                "class size is modest rather than decisive."
            )
        else:
            interpretation = (
                "Although the estimated coefficients are generally negative, they are "
                "not statistically distinguishable from zero at conventional levels, "
                "so the data do not provide strong evidence that lower "
                "student–teacher ratios are associated with higher academic "
                "performance once sampling variability is taken into account."
            )
    elif sign > 0:
        interpretation = (
            "The positive direction of the main coefficient suggests that, in this "
            "dataset, higher student–teacher ratios are associated with higher test "
            "scores after controlling for observed covariates, which runs counter to "
            "the hypothesized relationship."
        )
    else:
        interpretation = (
            "The estimates do not show a clear directional effect of the "
            "student–teacher ratio on academic performance."
        )

    explanation_lines.append(interpretation)
    explanation_lines.append(
        f"On a 0–100 Likert scale, where 0 represents a strong 'No' and 100 a strong "
        f"'Yes' to the research question, I assign a response of {response:d}, "
        "reflecting the strength and direction of the statistical evidence in this "
        "dataset."
    )

    explanation = " ".join(explanation_lines)

    result = {"response": int(response), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        # Single JSON object, no extra lines or text
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
