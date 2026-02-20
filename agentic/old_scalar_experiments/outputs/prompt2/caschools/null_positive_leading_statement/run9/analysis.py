import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("caschools.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    df = df.dropna(subset=["stratio", "testscr"])
    n = int(df.shape[0])

    # Simple bivariate association
    corr = df["stratio"].corr(df["testscr"])

    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    beta1 = float(model1.params["stratio"])
    p1 = float(model1.pvalues["stratio"])

    # Multiple regression with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks"]
    available_controls = [c for c in controls if c in df.columns]

    if available_controls:
        X2 = sm.add_constant(df[["stratio"] + available_controls].dropna())
        y2 = df.loc[X2.index, "testscr"]
        model2 = sm.OLS(y2, X2).fit()
        beta2 = float(model2.params["stratio"])
        p2 = float(model2.pvalues["stratio"])
    else:
        # Fallback: no controls available
        beta2 = beta1
        p2 = p1

    # Decision rule for the research question
    evidence_negative = (beta1 < 0) and (beta2 < 0)
    evidence_significant = (p1 < 0.05) or (p2 < 0.05)

    if evidence_negative and evidence_significant:
        response = "Yes"
        confidence = 85
    elif evidence_negative:
        response = "Yes"
        confidence = 70
    else:
        response = "No"
        confidence = 70 if (beta1 > 0 and beta2 > 0) else 55

    explanation_parts = [
        f"Research question: {question}",
        f"I computed the student-teacher ratio as students divided by teachers "
        f"and the academic performance measure as the average of reading and "
        f"math scores for each district (N={n}).",
        f"The Pearson correlation between student-teacher ratio and average "
        f"test score is {corr:.3f}, indicating that districts with lower ratios "
        f"tend to have higher scores when this value is negative.",
        f"A simple OLS regression of average test score on the student-teacher "
        f"ratio yields a coefficient of {beta1:.3f} (p-value={p1:.3f}).",
        f"A multiple regression controlling for income, percent English learners, "
        f"percent qualifying for reduced-price lunch, and percent on CalWorks "
        f"produces a coefficient of {beta2:.3f} (p-value={p2:.3f}).",
    ]

    if response == "Yes":
        explanation_parts.append(
            "In both models, the coefficient on the student-teacher ratio is "
            "negative, meaning that lower ratios (smaller classes) are "
            "associated with higher average test scores. These consistent "
            "negative associations support the claim that a lower "
            "student-teacher ratio is associated with higher academic "
            "performance, although the effect size is modest and the "
            "observational nature of the data limits strong causal "
            "interpretation."
        )
    else:
        explanation_parts.append(
            "The estimated relationship is not consistently negative and "
            "statistically robust across specifications, so the data do not "
            "provide clear evidence that lower student-teacher ratios are "
            "associated with higher academic performance. The observational "
            "design also limits strong causal interpretation."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

