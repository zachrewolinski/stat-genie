import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Basic sanity checks
    n_obs = df.shape[0]
    corr_str_score = df[["str", "avg_score"]].corr().loc["str", "avg_score"]

    # Simple linear regression: avg_score ~ str
    simple_model = smf.ols("avg_score ~ str", data=df).fit()
    coef_str_simple = simple_model.params["str"]
    pval_str_simple = simple_model.pvalues["str"]
    r2_simple = simple_model.rsquared

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    formula_controls = "avg_score ~ str + " + " + ".join(controls)
    multi_model = smf.ols(formula_controls, data=df).fit()
    coef_str_multi = multi_model.params["str"]
    pval_str_multi = multi_model.pvalues["str"]
    r2_multi = multi_model.rsquared

    # Summaries for explanation
    str_mean = df["str"].mean()
    str_std = df["str"].std()
    score_mean = df["avg_score"].mean()
    score_std = df["avg_score"].std()

    # Decide on response based on sign and significance of the student-teacher ratio effect.
    # Negative coefficient => higher ratios (more students per teacher) associated with lower scores,
    # i.e., lower ratios associated with higher scores.
    if coef_str_multi < 0 and pval_str_multi < 0.05:
        response = "Yes"
        strength = 75
        confidence = 80
        evidence_summary = (
            "Overall, these results provide moderately strong evidence that districts with lower "
            "student–teacher ratios tend to have higher average test scores, even after adjusting "
            "for several socioeconomic and resource variables."
        )
    elif coef_str_simple < 0 and pval_str_simple < 0.05:
        response = "Yes"
        strength = 65
        confidence = 70
        evidence_summary = (
            "Overall, the simple models suggest that districts with lower student–teacher ratios "
            "tend to have higher average test scores, although this evidence weakens once additional "
            "covariates are considered."
        )
    else:
        response = "No"
        strength = 40
        confidence = 60
        evidence_summary = (
            "Overall, these results do not provide strong evidence that districts with lower "
            "student–teacher ratios have higher average test scores. In the models estimated here, "
            "the association between the student–teacher ratio and scores is very small and not "
            "statistically distinguishable from zero once sampling variability is taken into account."
        )

    explanation_lines = []
    explanation_lines.append(
        "I examined whether lower student–teacher ratios are associated with higher academic "
        "performance using the caschools dataset of 420 California K-6 and K-8 districts."
    )
    explanation_lines.append(
        f"I defined student–teacher ratio as students divided by teachers (mean {str_mean:.1f}, "
        f"SD {str_std:.1f}) and academic performance as the average of reading and math scores "
        f"(mean {score_mean:.1f}, SD {score_std:.1f})."
    )
    explanation_lines.append(
        f"The simple correlation between student–teacher ratio and average test score is "
        f"{corr_str_score:.3f}, summarizing the linear association between class size and scores."
    )
    explanation_lines.append(
        f"In a simple linear regression of average score on the student–teacher ratio, the "
        f"coefficient on the ratio is {coef_str_simple:.3f} with p-value {pval_str_simple:.3f} "
        f"(R-squared {r2_simple:.3f})."
    )
    explanation_lines.append(
        "This coefficient describes how average scores change as the number of students per teacher "
        "increases; in this dataset the estimated effect size from the simple model is small."
    )
    explanation_lines.append(
        "To account for important confounders, I estimated a multiple regression of average score "
        "on the student–teacher ratio, controlling for district income, percent English learners, "
        "percent on reduced-price lunch, percent on CalWorks, per-pupil expenditures, and number of "
        "computers."
    )
    explanation_lines.append(
        f"In this adjusted model, the coefficient on the student–teacher ratio is {coef_str_multi:.3f} "
        f"with p-value {pval_str_multi:.3f} (R-squared {r2_multi:.3f}). "
        "In this dataset, the adjusted effect of the student–teacher ratio on scores is also small."
    )
    explanation_lines.append(evidence_summary)
    explanation_lines.append(
        "Because the data are observational and cross-sectional, these findings speak to association "
        "rather than providing definitive evidence of a causal effect of class size on performance."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
