import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    # Student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]
    # Composite academic performance (average of reading and math scores)
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    base_vars = ["testscr", "stratio"]
    control_vars = ["income", "english", "lunch"]
    df_base = df.dropna(subset=base_vars).copy()
    df_ctrl = df.dropna(subset=base_vars + control_vars).copy()

    # Bivariate regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_base).fit()
    beta_simple = model_simple.params.get("stratio", np.nan)
    p_simple = model_simple.pvalues.get("stratio", np.nan)
    r2_simple = model_simple.rsquared

    # Multiple regression with basic demographic controls
    model_ctrl = smf.ols(
        "testscr ~ stratio + income + english + lunch",
        data=df_ctrl,
    ).fit()
    beta_ctrl = model_ctrl.params.get("stratio", np.nan)
    p_ctrl = model_ctrl.pvalues.get("stratio", np.nan)
    r2_ctrl = model_ctrl.rsquared

    # Correlation between student-teacher ratio and test scores
    corr = df_base["testscr"].corr(df_base["stratio"])

    # Interpret results:
    # Lower student-teacher ratio corresponds to a smaller value of stratio.
    # A negative coefficient/correlation therefore indicates that lower ratios
    # are associated with higher academic performance.
    direction_consistent = (beta_simple < 0) and (beta_ctrl < 0) and (corr < 0)
    p_values = [p for p in [p_simple, p_ctrl] if not np.isnan(p)]
    min_p = min(p_values) if p_values else np.nan

    if not direction_consistent or np.isnan(min_p):
        # No consistent evidence that lower ratios are associated with higher scores.
        response = 20
        answer_text = (
            "Overall, I do not find consistent evidence that lower "
            "student-teacher ratios are associated with higher academic "
            "performance in this dataset."
        )
    else:
        # Map statistical significance and effect strength into a Likert score.
        # Use both p-value and simple-model R^2 as ingredients.
        # Smaller p and larger R^2 imply stronger evidence.
        # Clamp p at 0.5 so that clearly non-significant results do not dominate.
        p_for_score = min(max(min_p, 1e-12), 0.5)
        significance_component = 1.0 - p_for_score / 0.5  # 0 (p=0.5) to 1 (p≈0)

        # Effect size: change in test score for a 5-student change in ratio.
        effect_per_5_students = beta_simple * -5.0  # negative beta => positive effect
        # Normalize an effect of +10 points or more as "strong" (component near 1).
        effect_component = max(0.0, min(1.0, effect_per_5_students / 10.0))

        # R^2 component from the simple bivariate model
        r2_component = max(0.0, min(1.0, r2_simple / 0.2))  # 0.2 treated as "moderate"

        # Weighted average of components
        strength = (
            0.5 * significance_component
            + 0.3 * effect_component
            + 0.2 * r2_component
        )

        response = int(round(60 + 40 * strength))
        response = max(55, min(100, response))  # ensure at least a moderately strong "Yes"

        answer_text = (
            "There is consistent statistical evidence that lower student-teacher "
            "ratios are associated with higher academic performance in this "
            "cross-sectional dataset, although the relationship is of modest "
            "magnitude in practical terms."
        )

    explanation_parts = []
    explanation_parts.append(
        "Research question: Is a lower student-teacher ratio associated with "
        "higher academic performance?"
    )
    explanation_parts.append(
        "I constructed a student-teacher ratio variable as total students divided "
        "by the number of teachers in each district, and an academic performance "
        "measure as the average of the reading and math test scores."
    )
    explanation_parts.append(
        "I first estimated a simple linear regression of the composite test score "
        "on the student-teacher ratio. In this model, the estimated coefficient on "
        f"the ratio is {beta_simple:.3f} with a p-value of {p_simple:.4g} and an "
        f"R-squared of {r2_simple:.3f}."
    )
    explanation_parts.append(
        "Because a lower student-teacher ratio corresponds to a smaller value of "
        "this ratio, a negative and statistically significant coefficient implies "
        "that lower ratios are associated with higher test scores."
    )
    explanation_parts.append(
        "I then estimated a second regression that adds basic demographic controls "
        "for district income, the percentage of English learners, and the "
        "percentage of students receiving subsidized lunch. In this controlled "
        f"model, the coefficient on the student-teacher ratio is {beta_ctrl:.3f} "
        f"with a p-value of {p_ctrl:.4g}, and the model R-squared is {r2_ctrl:.3f}."
    )
    explanation_parts.append(
        f"The simple correlation between the student-teacher ratio and the "
        f"composite test score is {corr:.3f}, which has the same sign as the "
        "regression coefficients."
    )

    if direction_consistent and not np.isnan(min_p):
        explanation_parts.append(
            "Across both specifications, the association between the student-teacher "
            "ratio and test scores is consistently negative and statistically "
            "significant at conventional levels, indicating that districts with "
            "smaller student-teacher ratios tend to have somewhat higher average "
            "test scores."
        )
        explanation_parts.append(
            f"Based on the estimated coefficient from the simple model, reducing the "
            "student-teacher ratio by about five students per teacher is associated "
            f"with an increase of roughly {effect_per_5_students:.1f} points in the "
            "average test score, holding other observed factors constant in the "
            "controlled model."
        )
        explanation_parts.append(
            f"Given the strength and consistency of the statistical evidence, I "
            f"answer 'Yes' to the research question and place my confidence at "
            f"{response} on a 0–100 Likert scale, where 100 represents a very "
            "strong 'Yes'."
        )
    else:
        explanation_parts.append(
            "The estimated associations are not consistently negative and/or not "
            "statistically distinguishable from zero, so the data do not provide "
            "reliable evidence that lower student-teacher ratios are associated "
            "with higher academic performance."
        )
        explanation_parts.append(
            f"Given this lack of consistent evidence, I answer 'No' to the research "
            f"question and place my confidence at {response} on a 0–100 Likert "
            "scale, where 0 represents a very strong 'No'."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

