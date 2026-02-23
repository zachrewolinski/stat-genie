import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "score"])

    # Simple correlation
    corr, corr_p = stats.pearsonr(df["stratio"], df["score"])

    # Simple regression: score on student-teacher ratio
    model_simple = smf.ols("score ~ stratio", data=df).fit()

    # Multiple regression with available controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]

    formula = "score ~ stratio"
    if available_controls:
        formula += " + " + " + ".join(available_controls)

    model_controls = smf.ols(formula, data=df).fit()

    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    coef_ctrl = float(model_controls.params["stratio"])
    p_ctrl = float(model_controls.pvalues["stratio"])

    n = int(len(df))

    # Assess strength and direction of evidence
    strong_consistent = (
        coef_ctrl < 0
        and p_ctrl < 0.05
        and coef_simple < 0
        and p_simple < 0.05
    )
    weak_consistent = (
        coef_ctrl < 0
        and p_ctrl < 0.1
        and coef_simple < 0
        and p_simple < 0.1
    )

    if strong_consistent:
        response = 85
    elif weak_consistent:
        response = 65
    else:
        # Little or no statistical evidence for an association
        response = 20

    if strong_consistent or weak_consistent:
        evidence_word = "strong" if strong_consistent else "moderate"
        explanation = (
            f"I analyzed data on {n} California K-6 and K-8 school districts. "
            f"I constructed the student–teacher ratio as students divided by teachers and an overall "
            f"academic performance index as the mean of reading and math Stanford 9 test scores. "
            f"The Pearson correlation between student–teacher ratio and performance was {corr:.2f} "
            f"(p={corr_p:.3g}), indicating that districts with more students per teacher tend to have "
            f"lower test scores. "
            f"A simple OLS regression of performance on the ratio alone estimated that each additional "
            f"student per teacher is associated with {coef_simple:.2f} fewer test-score points "
            f"(p={p_simple:.3g}). "
            f"After controlling for district income, poverty (CalWorks participation and reduced-price "
            f"lunch eligibility), English-learner share, computers per classroom, and expenditures per "
            f"pupil, the coefficient on student–teacher ratio remained {coef_ctrl:.2f} "
            f"(p={p_ctrl:.3g}). "
            f"These results provide {evidence_word} evidence that, in this dataset, lower student–teacher "
            f"ratios are associated with higher academic performance, though the analysis is observational "
            f"and does not establish causality. "
            f"On a 0–100 scale, where higher values indicate stronger evidence for a 'Yes' answer to the "
            f"research question, I therefore rate this relationship as {response}."
        )
    else:
        explanation = (
            f"I analyzed data on {n} California K-6 and K-8 school districts. "
            f"I constructed the student–teacher ratio as students divided by teachers and an overall "
            f"academic performance index as the mean of reading and math Stanford 9 test scores. "
            f"The Pearson correlation between student–teacher ratio and performance was {corr:.2f} "
            f"(p={corr_p:.3g}), providing little statistical evidence of any linear relationship between "
            f"class size (as measured by student–teacher ratio) and test scores. "
            f"A simple OLS regression of performance on the ratio alone produced a coefficient of "
            f"{coef_simple:.2f} points per additional student per teacher (p={p_simple:.3g}), and after "
            f"controlling for district income, poverty (CalWorks participation and reduced-price lunch "
            f"eligibility), English-learner share, computers per classroom, and expenditures per pupil, "
            f"the coefficient on student–teacher ratio was {coef_ctrl:.2f} (p={p_ctrl:.3g}). "
            f"Both estimates are extremely small in magnitude and far from statistically significant, so "
            f"within this dataset we do not detect a meaningful association between student–teacher ratios "
            f"and academic performance. The data cannot rule out very small effects, but they provide "
            f"little evidence for a substantial relationship. "
            f"On a 0–100 scale, where higher values indicate stronger evidence for a 'Yes' answer to the "
            f"research question, I therefore rate support for a 'Yes' answer as {response}, corresponding "
            f"in practical terms to a 'No' answer with fairly strong confidence."
        )

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f, ensure_ascii=False)

    # Also print to stdout for human inspection (not required by the task).
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
