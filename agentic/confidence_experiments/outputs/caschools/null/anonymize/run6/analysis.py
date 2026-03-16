import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Construct student–teacher ratio and average test score.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_testscr"] = (df["feature14"] + df["feature15"]) / 2.0
    return df


def fit_models(df: pd.DataFrame):
    df = df.dropna(subset=["student_teacher_ratio", "avg_testscr"])

    # Simple bivariate regression: test score on student–teacher ratio.
    X1 = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_testscr"], X1).fit()

    # Multiple regression controlling for key covariates:
    # - feature8: % CalWorks (income assistance)
    # - feature9: % reduced-price lunch
    # - feature10: number of computers
    # - feature11: expenditure per student
    # - feature12: district average income
    # - feature13: % English learners
    covariates = [
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
    available_covars = [c for c in covariates if c in df.columns]
    X2 = sm.add_constant(df[["student_teacher_ratio"] + available_covars])
    model_controls = sm.OLS(df["avg_testscr"], X2).fit()

    return model_simple, model_controls


def summarize_relationship(df: pd.DataFrame, model_simple, model_controls):
    # Extract key statistics from both models.
    coef1 = float(model_simple.params["student_teacher_ratio"])
    pval1 = float(model_simple.pvalues["student_teacher_ratio"])
    ci1_low, ci1_high = model_simple.conf_int().loc["student_teacher_ratio"]
    r2_1 = float(model_simple.rsquared)

    coef2 = float(model_controls.params["student_teacher_ratio"])
    pval2 = float(model_controls.pvalues["student_teacher_ratio"])
    ci2_low, ci2_high = model_controls.conf_int().loc["student_teacher_ratio"]
    r2_2 = float(model_controls.rsquared)

    # Correlation for effect size intuition.
    corr = float(df["student_teacher_ratio"].corr(df["avg_testscr"]))

    stats = {
        "coef_simple": coef1,
        "pval_simple": pval1,
        "ci_simple": (float(ci1_low), float(ci1_high)),
        "r2_simple": r2_1,
        "coef_controls": coef2,
        "pval_controls": pval2,
        "ci_controls": (float(ci2_low), float(ci2_high)),
        "r2_controls": r2_2,
        "corr": corr,
    }
    return stats


def compute_likert_response(stats: dict) -> int:
    coef1 = stats["coef_simple"]
    p1 = stats["pval_simple"]
    coef2 = stats["coef_controls"]
    p2 = stats["pval_controls"]
    corr = stats["corr"]

    # We are testing whether LOWER student–teacher ratios
    # (fewer students per teacher) are associated with HIGHER scores.
    # That corresponds to a NEGATIVE coefficient/correlation.
    strong_negative = (coef1 < 0) and (coef2 < 0)
    sig_strong = (p1 < 0.01) and (p2 < 0.01)
    sig_moderate = (p1 < 0.05) and (p2 < 0.05)

    if strong_negative and sig_strong:
        base = 85
    elif strong_negative and sig_moderate:
        base = 75
    elif coef1 < 0 and p1 < 0.05:
        base = 65
    elif coef1 < 0 and p1 < 0.1:
        base = 55
    else:
        # Little or no evidence for the hypothesized direction.
        base = 40 if coef1 < 0 else 30

    # Adjust based on the strength of the linear association.
    abs_corr = abs(corr)
    if abs_corr > 0.4:
        base += 10
    elif abs_corr > 0.25:
        base += 5
    elif abs_corr < 0.1:
        base -= 5

    # Clip to the required 0–100 integer range.
    response = int(round(max(0, min(100, base))))
    return response


def build_explanation(stats: dict, response: int) -> str:
    coef1 = stats["coef_simple"]
    p1 = stats["pval_simple"]
    ci1_low, ci1_high = stats["ci_simple"]
    r2_1 = stats["r2_simple"]

    coef2 = stats["coef_controls"]
    p2 = stats["pval_controls"]
    ci2_low, ci2_high = stats["ci_controls"]
    r2_2 = stats["r2_controls"]

    corr = stats["corr"]

    direction = "negative" if coef1 < 0 else "positive"
    support_phrase = "support" if response >= 50 else "do not support"

    # Describe correlation strength.
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        corr_desc = "essentially no linear association"
    elif abs_corr < 0.3:
        corr_desc = "a weak linear association"
    elif abs_corr < 0.5:
        corr_desc = "a moderate linear association"
    else:
        corr_desc = "a strong linear association"

    # Describe statistical significance across models.
    both_sig = (p1 < 0.05) and (p2 < 0.05)
    any_sig = (p1 < 0.05) or (p2 < 0.05)
    if both_sig:
        sig_sentence = (
            "In both models, this association is statistically significant at conventional levels."
        )
    elif any_sig:
        sig_sentence = (
            "This association is statistically significant in one of the models but not the other."
        )
    else:
        sig_sentence = (
            "In neither model is this association statistically significant at conventional levels."
        )

    # Describe whether the estimated direction matches the hypothesis.
    in_expected_direction = (coef1 < 0) and (coef2 < 0)
    if in_expected_direction:
        direction_sentence = (
            "The estimated coefficients are negative in both models, consistent with the hypothesis "
            "that smaller classes (lower student-teacher ratios) are associated with higher scores."
        )
    elif (coef1 < 0) or (coef2 < 0):
        direction_sentence = (
            "The estimated coefficients are negative in one model but not the other, offering only mixed "
            "support for the hypothesized direction."
        )
    else:
        direction_sentence = (
            "The estimated coefficients are non-negative, providing no evidence that smaller classes are "
            "associated with higher scores."
        )

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n\n"
        "Using data on 420 California school districts, I constructed the student-teacher ratio as total "
        "enrollment divided by the number of teachers and measured academic performance as the average of "
        "district reading and math scores.\n\n"
        f"In a simple linear regression of average test score on the student-teacher ratio, each additional "
        f"student per teacher is associated with a {coef1:.2f}-point change in the average test score "
        f"({direction} association; p = {p1:.4g}, 95% CI [{ci1_low:.2f}, {ci1_high:.2f}], R² = {r2_1:.3f}). "
        "Because lower ratios correspond to fewer students per teacher, a negative coefficient means that "
        "districts with smaller classes tend to have higher test scores.\n\n"
        "To account for potential confounding, I estimated a second regression that controls for district "
        "demographics and resources: the percentages of students in income-assistance (CalWorks) and reduced-price "
        "lunch programs, the number of computers, expenditure per student, average district income, and the percentage "
        "of English learners. In this model, the coefficient on the student-teacher ratio remains "
        f"{coef2:.2f} (p = {p2:.4g}, 95% CI [{ci2_low:.2f}, {ci2_high:.2f}], R² = {r2_2:.3f}).\n\n"
        f"The Pearson correlation between the student-teacher ratio and average test scores is {corr:.3f}, indicating {corr_desc}. "
        f"{sig_sentence} {direction_sentence}\n\n"
        f"Overall, these results {support_phrase} the claim that lower student-teacher ratios are associated with higher "
        "academic performance. The Likert-scale response of "
        f"{response} (0 = strong 'No', 100 = strong 'Yes') reflects this evidence."
    )

    return explanation


def main():
    df = load_data("caschools.csv")
    model_simple, model_controls = fit_models(df)
    stats = summarize_relationship(df, model_simple, model_controls)
    response = compute_likert_response(stats)
    explanation = build_explanation(stats, response)

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON object to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
