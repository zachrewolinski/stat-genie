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
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (defensive; dataset is mostly complete)
    key_cols = ["stratio", "testscr", "income", "english", "lunch", "calworks"]
    df_model = df.dropna(subset=key_cols).copy()

    # Simple correlation
    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key socioeconomic controls
    formula_controls = "testscr ~ stratio + income + english + lunch + calworks"
    model_controls = smf.ols(formula_controls, data=df_model).fit()
    coef_controls = float(model_controls.params["stratio"])
    pval_controls = float(model_controls.pvalues["stratio"])
    r2_controls = float(model_controls.rsquared)

    # Effect size: predicted change for a 5-student change in ratio (using controlled model)
    delta_ratio = 5.0
    effect_5 = coef_controls * (-delta_ratio)  # Lower ratio (smaller classes) should increase testscr if coef < 0

    # Decide Likert-scale response (0-100, 0=strong No, 100=strong Yes)
    # Heuristic: strong evidence if negative and p<0.01 in controlled model; moderate if p<0.05.
    if coef_controls < 0 and pval_controls < 0.01:
        response = 85
    elif coef_controls < 0 and pval_controls < 0.05:
        response = 70
    elif coef_controls < 0:
        response = 60
    elif coef_controls > 0 and pval_controls < 0.05:
        response = 25
    else:
        response = 50

    # Clip to valid bounds and ensure int
    response_int = int(np.clip(response, 0, 100))

    # Build explanation string with key numeric evidence
    explanation_lines = []
    explanation_lines.append(
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "in California K–6 and K–8 districts?"
    )
    explanation_lines.append(
        "I constructed the student–teacher ratio as students/teachers and an overall test score as the average "
        "of reading and math scores for each district."
    )
    explanation_lines.append(
        f"The simple Pearson correlation between student–teacher ratio and average test score is {corr:.3f}, "
        "which indicates the direction and strength of their linear association."
    )
    explanation_lines.append(
        "I first fit an ordinary least squares regression of average test score on student–teacher ratio alone. "
        f"In this simple model, the coefficient on the ratio is {coef_simple:.3f} with p-value {pval_simple:.3g} "
        f"and R-squared {r2_simple:.3f}."
    )
    explanation_lines.append(
        "To account for observable socioeconomic differences across districts, I then fit a multiple regression "
        "of average test score on student–teacher ratio while controlling for average district income, the "
        "percentage of students on public assistance (CalWorks), the percentage eligible for reduced-price lunch, "
        "and the percentage of English learners."
    )
    explanation_lines.append(
        f"In this controlled model, the coefficient on student–teacher ratio is {coef_controls:.3f} with "
        f"p-value {pval_controls:.3g} and R-squared {r2_controls:.3f}."
    )
    explanation_lines.append(
        f"Interpreting this coefficient, a reduction of 5 students per teacher is associated with an estimated "
        f"change of {effect_5:.2f} points in the average test score (holding the included socioeconomic variables "
        "constant)."
    )
    explanation_lines.append(
        "Because the estimated association between student–teacher ratio and test scores is "
        f"{'negative' if coef_controls < 0 else 'non-negative'} and the corresponding p-value in the controlled "
        "model is "
        f"{'below' if pval_controls < 0.05 else 'not below'} the conventional 0.05 threshold, "
        "the data provide "
        f"{'moderate to strong' if coef_controls < 0 and pval_controls < 0.05 else 'limited'} statistical evidence "
        "that districts with smaller student–teacher ratios tend to have higher average test scores, "
        "after adjusting for key socioeconomic characteristics. However, the data are observational, so this "
        "analysis supports an association rather than a definitive causal effect."
    )
    explanation_lines.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', I summarize this evidence "
        f"with a response of {response_int}, reflecting that the dataset suggests a "
        f"{'meaningful' if response_int >= 70 else 'somewhat positive' if response_int > 50 else 'weak'} "
        "association between lower student–teacher ratios and higher academic performance, "
        "but with some remaining uncertainty due to potential unobserved confounding and model assumptions."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response_int,
        "explanation": explanation,
    }

    # Write required JSON output; file must contain only this JSON object.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

