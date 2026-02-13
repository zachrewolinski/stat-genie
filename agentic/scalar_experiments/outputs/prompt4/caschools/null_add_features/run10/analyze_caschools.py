import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Core variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the key variables
    core_vars = ["stratio", "testscr", "income", "english", "lunch"]
    df_core = df.dropna(subset=core_vars)

    # Simple Pearson correlation between student-teacher ratio and test scores
    r, p_value = stats.pearsonr(df_core["stratio"], df_core["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_core["stratio"])
    model_simple = sm.OLS(df_core["testscr"], X_simple).fit()
    beta_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key demographics:
    # testscr ~ stratio + income + english + lunch
    X_multi = df_core[["stratio", "income", "english", "lunch"]]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_core["testscr"], X_multi).fit()
    beta_multi = model_multi.params["stratio"]
    p_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    # Map statistical evidence to a 0-100 Likert response.
    # Direction: negative beta / correlation means lower ratio -> higher scores.
    evidence_score = 50.0

    # Use correlation magnitude as a baseline (cap at |r| = 0.5).
    corr_strength = min(abs(r), 0.5) / 0.5  # in [0,1]

    if beta_simple < 0 and r < 0:
        # Association in hypothesized direction.
        evidence_score = 60 + 25 * corr_strength
    else:
        # Association not in hypothesized direction.
        evidence_score = 40 - 25 * corr_strength

    # Strengthen evidence based on p-values from simple and multiple regression.
    # Smaller p-values push further from 50 toward the corresponding direction.
    def p_to_bonus(p: float) -> float:
        if p < 0.001:
            return 10.0
        if p < 0.01:
            return 7.0
        if p < 0.05:
            return 4.0
        if p < 0.1:
            return 2.0
        return 0.0

    direction = 1.0 if beta_simple < 0 else -1.0
    bonus = p_to_bonus(p_simple) + p_to_bonus(p_multi)
    evidence_score += direction * bonus

    # Clip to [0, 100] and convert to an integer.
    response_value = int(np.clip(round(evidence_score), 0, 100))

    explanation_parts = []
    explanation_parts.append(
        "Using the caschools.csv dataset of 420 California K-6 and K-8 districts, "
        "I examined whether lower student-teacher ratios are associated with higher academic performance."
    )
    explanation_parts.append(
        "I defined the student-teacher ratio as the number of students divided by the number of teachers "
        "and measured academic performance as the average of the district-level reading and math Stanford 9 test scores."
    )

    # Describe the correlation in a way that reflects its magnitude and significance.
    if abs(r) < 0.05 or p_value >= 0.05:
        explanation_parts.append(
            f"The Pearson correlation between the student-teacher ratio and average test score was {r:.3f} "
            f"(p-value = {p_value:.3g}), providing little evidence of any meaningful linear relationship between "
            f"class size and test performance."
        )
    else:
        if r < 0:
            corr_direction = (
                "a negative correlation, so districts with more students per teacher tend to have lower scores "
                "while lower ratios are associated with higher scores."
            )
        else:
            corr_direction = (
                "a positive correlation, so districts with more students per teacher tend to have higher scores "
                "while lower ratios are associated with slightly lower scores."
            )
        explanation_parts.append(
            f"The Pearson correlation between the student-teacher ratio and average test score was {r:.3f} "
            f"(p-value = {p_value:.3g}), indicating {corr_direction}"
        )

    # Simple regression interpretation
    simple_signif = "statistically significant" if p_simple < 0.05 else "not statistically significant"
    if beta_simple < 0:
        simple_direction = (
            "higher student-teacher ratios are associated with lower average test scores, consistent with smaller "
            "class sizes being beneficial"
        )
    elif beta_simple > 0:
        simple_direction = (
            "higher student-teacher ratios are associated with slightly higher average test scores, which goes against "
            "the expectation that smaller classes help"
        )
    else:
        simple_direction = (
            "there is essentially no change in average test scores as the student-teacher ratio varies"
        )

    explanation_parts.append(
        f"In a simple linear regression of test scores on the student-teacher ratio, the estimated coefficient on the ratio "
        f"was {beta_simple:.3f} with p-value {p_simple:.3g} and R-squared {r2_simple:.3f}; this means that, on average, "
        f"each additional student per teacher is associated with a {beta_simple:.3f}-point change in the test score, so "
        f"{simple_direction}, but this estimated effect is {simple_signif} and explains very little of the variation in scores."
    )

    # Multiple regression interpretation
    multi_signif = "statistically significant" if p_multi < 0.05 else "not statistically significant"
    if beta_multi < 0:
        multi_direction = "a negative association after adjustment"
    elif beta_multi > 0:
        multi_direction = "a positive association after adjustment"
    else:
        multi_direction = "essentially no association after adjustment"

    explanation_parts.append(
        f"When adding controls for district income, percentage of English learners, and percentage of students eligible "
        f"for reduced-price lunch, the coefficient on the student-teacher ratio was {beta_multi:.3f} with p-value "
        f"{p_multi:.3g} and model R-squared {r2_multi:.3f}, indicating {multi_direction} between the ratio and test scores "
        f"that is {multi_signif}."
    )

    if response_value >= 67:
        overall_sentence = (
            "Overall, these results provide strong statistical evidence that lower student-teacher ratios are associated "
            "with higher academic performance in this dataset."
        )
    elif response_value <= 33:
        overall_sentence = (
            "Overall, these results provide strong statistical evidence against a substantial benefit of lower "
            "student-teacher ratios for academic performance in this dataset."
        )
    else:
        overall_sentence = (
            "Overall, these results provide little clear evidence that student-teacher ratios are meaningfully associated "
            "with academic performance in this dataset."
        )

    explanation_parts.append(overall_sentence)

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
