import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map anonymized feature names to their documented meanings.
    enrollment = df["feature6"]
    teachers = df["feature7"]
    calworks_pct = df["feature8"]
    lunch_pct = df["feature9"]
    income = df["feature12"]
    english_learners_pct = df["feature13"]
    read_score = df["feature14"]
    math_score = df["feature15"]

    # Construct key variables.
    # Student-teacher ratio: students per teacher (higher = more students per teacher).
    stratio = enrollment / teachers
    stratio.name = "stratio"
    # Overall academic performance: average of reading and math scores.
    testscr = (read_score + math_score) / 2.0

    # Simple correlation between ratio and performance.
    corr = np.corrcoef(stratio, testscr)[0, 1]

    # Simple linear regression: testscr ~ stratio.
    X_simple = sm.add_constant(stratio)
    model_simple = sm.OLS(testscr, X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key demographics.
    X_controls = pd.DataFrame(
        {
            "stratio": stratio,
            "calworks": calworks_pct,
            "lunch": lunch_pct,
            "income": income,
            "english": english_learners_pct,
        }
    )
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(testscr, X_controls).fit()
    coef_controls = model_controls.params["stratio"]
    pval_controls = model_controls.pvalues["stratio"]

    # Determine answer logic.
    # Evidence for "Yes" (lower ratio -> higher scores) requires
    # consistently negative association across correlation and regressions.
    negative_corr = corr < 0
    negative_coef_simple = coef_simple < 0
    negative_coef_controls = coef_controls < 0

    yes_evidence = 0
    if negative_corr:
        yes_evidence += 20
    if negative_coef_simple:
        yes_evidence += 30
    if negative_coef_controls:
        yes_evidence += 30
    if pval_simple < 0.05 and negative_coef_simple:
        yes_evidence += 10
    if pval_controls < 0.05 and negative_coef_controls:
        yes_evidence += 10

    yes_evidence = max(0, min(100, yes_evidence))

    if (
        negative_corr
        and negative_coef_simple
        and negative_coef_controls
        and pval_simple < 0.1
        and pval_controls < 0.1
    ):
        response = "Yes"
        strength = int(yes_evidence if yes_evidence > 0 else 60)
    else:
        response = "No"
        # For a "No" answer, we are measuring how strongly the
        # data fail to show a meaningful negative association.
        strength = 50
        if abs(corr) < 0.05:
            strength += 15
        if pval_simple > 0.1:
            strength += 15
        if pval_controls > 0.1:
            strength += 15
        if (coef_simple >= 0 and coef_controls >= 0) or (coef_simple <= 0 and coef_controls <= 0):
            strength += 5
        strength = max(0, min(100, strength))

    # Confidence slightly lower than strength to reflect modeling choices.
    confidence = max(0, min(100, int(strength * 0.9)))

    # Build explanation text with interpretation that depends on direction and significance.
    if abs(corr) < 0.05:
        corr_sentence = (
            f"The Pearson correlation between student-teacher ratio and average test score was {corr:.3f}, "
            "very close to zero, indicating essentially no linear relationship between the two variables."
        )
    elif corr < 0:
        corr_sentence = (
            f"The Pearson correlation between student-teacher ratio and average test score was {corr:.3f}, "
            "a negative association consistent with lower ratios being linked to higher scores."
        )
    else:
        corr_sentence = (
            f"The Pearson correlation between student-teacher ratio and average test score was {corr:.3f}, "
            "a positive association indicating that districts with more students per teacher tended, if anything, "
            "to have slightly higher scores."
        )

    direction_simple = "higher" if coef_simple > 0 else "lower"
    signif_simple = (
        "statistically significant"
        if pval_simple < 0.05
        else "not statistically distinguishable from zero at conventional levels"
    )
    simple_sentence = (
        "In a simple OLS regression of average test score on the student-teacher ratio, "
        f"the coefficient on the ratio was {coef_simple:.3f} (p-value {pval_simple:.3g}), meaning that, on average, "
        f"districts with more students per teacher tended to have {direction_simple} test scores; "
        f"however, this estimate is {signif_simple}."
    )

    direction_controls = "higher" if coef_controls > 0 else "lower"
    signif_controls = (
        "statistically significant"
        if pval_controls < 0.05
        else "not statistically distinguishable from zero at conventional levels"
    )
    controls_sentence = (
        "In a multiple regression controlling for CalWorks percentage, reduced-price-lunch percentage, "
        f"district income, and percentage of English learners, the coefficient on the ratio was {coef_controls:.3f} "
        f"(p-value {pval_controls:.3g}), implying that, holding these factors fixed, districts with more students per "
        f"teacher tended to have {direction_controls} test scores; again, this estimate is {signif_controls}."
    )

    if response == "Yes":
        summary_sentence = (
            "Taken together, the negative correlation and regression coefficients provide fairly consistent evidence "
            "that lower student-teacher ratios are associated with higher academic performance in this dataset."
        )
    else:
        summary_sentence = (
            "Overall, across correlation and regression analyses, the estimates are very small in magnitude and not "
            "statistically significant, so the data do not provide evidence that lower student-teacher ratios are "
            "associated with higher academic performance."
        )

    explanation_parts = [
        "Analyzed 420 California K-6/K-8 districts using student-teacher ratio (enrollment divided by number of "
        "teachers) and the average of reading and math scores as the academic performance measure.",
        corr_sentence,
        simple_sentence,
        controls_sentence,
        summary_sentence,
    ]
    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
