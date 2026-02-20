import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and average test score
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation
    corr = df[["stratio", "testscr"]].corr().loc["stratio", "testscr"]

    # Simple bivariate OLS: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_stratio_simple = model_simple.params["stratio"]
    pval_stratio_simple = model_simple.pvalues["stratio"]

    # Multivariate OLS controlling for key demographics and resources
    controls = ["income", "english", "lunch", "expenditure", "computer"]
    avail_controls = [c for c in controls if c in df.columns]
    X_controls = df[["stratio"] + avail_controls]
    X_controls = sm.add_constant(X_controls)
    model_full = sm.OLS(df["testscr"], X_controls).fit()
    coef_stratio_full = model_full.params["stratio"]
    pval_stratio_full = model_full.pvalues["stratio"]

    # Decide on evidence strength
    # Association is considered supported if coefficient on stratio is negative
    # (lower ratio -> higher performance) and statistically significant.
    assoc_supported = (coef_stratio_full < 0) and (pval_stratio_full < 0.05)

    # Map to a 0-100 Likert-style response for the question:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    # We answer "Yes" if assoc_supported, with strength based on effect size and p-value.
    if assoc_supported:
        # Normalize effect size roughly by test score SD to gauge magnitude
        testscr_sd = df["testscr"].std()
        # Effect of reducing class size by 5 students per teacher
        effect_5 = 5 * abs(coef_stratio_full)
        effect_sd_units = effect_5 / testscr_sd if testscr_sd > 0 else 0.0

        # Stronger evidence (larger standardized effect and small p-value) -> closer to 100
        # Clamp values into [0, 1]
        mag_score = min(effect_sd_units / 0.2, 1.0)  # around 0.2 SD considered moderate
        sig_score = min(-np.log10(pval_stratio_full + 1e-12) / 5.0, 1.0)
        strength = 0.5 * mag_score + 0.5 * sig_score
        response = int(round(70 + 30 * strength))  # between 70 and 100
        yes_no = "Yes"
    else:
        # Evidence against or inconclusive
        # Use similar scaling but leaning toward "No".
        testscr_sd = df["testscr"].std()
        effect_5 = 5 * coef_stratio_full
        effect_sd_units = effect_5 / testscr_sd if testscr_sd > 0 else 0.0
        mag_score = min(abs(effect_sd_units) / 0.2, 1.0)
        sig_score = min(-np.log10(pval_stratio_full + 1e-12) / 5.0, 1.0)
        strength = 0.5 * mag_score + 0.5 * sig_score

        if coef_stratio_full > 0 and pval_stratio_full < 0.05:
            # Statistically significant opposite-direction effect: strong "No".
            response = int(round(100 - (70 + 30 * strength)))
        else:
            # Inconclusive or very weak evidence: near the middle but leaning "No".
            response = int(round(35 - 25 * strength))
        response = max(0, min(100, response))
        yes_no = "No"

    # Build a human-readable explanation summarizing the key findings.
    lines = []
    lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
    )
    lines.append(
        f"I operationalized student-teacher ratio as STR = students/teachers and academic performance as the average of reading and math scores."
    )
    lines.append(
        f"The Pearson correlation between STR and average test score is {corr:.3f}, indicating that higher ratios are associated with {('lower' if corr < 0 else 'higher' if corr > 0 else 'no clear change')} performance."
    )
    lines.append(
        f"In a simple OLS regression of average test score on STR, the coefficient on STR is {coef_stratio_simple:.3f} with p-value {pval_stratio_simple:.3g}."
    )
    lines.append(
        "I then estimated a multiple regression of average test scores on STR while controlling for district income, percent English learners, percent on reduced-price lunch, per-pupil expenditure, and number of computers (where available)."
    )
    lines.append(
        f"In this full model, the coefficient on STR is {coef_stratio_full:.3f} with p-value {pval_stratio_full:.3g}."
    )

    if assoc_supported:
        lines.append(
            "The STR coefficient is negative and statistically significant at the 5% level, meaning that districts with smaller classes (lower STR) tend to have higher test scores even after adjusting for these covariates."
        )
    else:
        lines.append(
            "The STR coefficient is not negative and statistically significant at the 5% level, so the data do not provide robust evidence that smaller classes are associated with higher test scores once covariates are included."
        )

    if assoc_supported:
        conclusion = (
            f"Overall, the regression and correlation analyses indicate that lower student-teacher ratios are associated with higher academic performance in this dataset, so my answer to the research question is '{yes_no}'."
        )
    else:
        conclusion = (
            f"Overall, the regression and correlation analyses do not provide strong evidence that lower student-teacher ratios are associated with higher academic performance in this dataset, so my answer to the research question is '{yes_no}'."
        )

    lines.append(conclusion)

    explanation = " " .join(lines)

    output = {"response": int(response), "explanation": explanation}

    out_path = base / "conclusion.txt"
    out_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
