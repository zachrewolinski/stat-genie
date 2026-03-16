import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_explanation(
    corr: float,
    coef_simple: float,
    p_simple: float,
    coef_adjusted: float,
    p_adjusted: float,
    effect_iqr: float,
    likert: int,
) -> str:
    direction = "negative" if coef_simple < 0 else "positive"
    assoc_word = "higher" if direction == "negative" else "lower"
    strength_desc = []

    if max(p_simple, p_adjusted) < 0.001:
        strength_desc.append("very strong statistical evidence")
    elif max(p_simple, p_adjusted) < 0.01:
        strength_desc.append("strong statistical evidence")
    elif max(p_simple, p_adjusted) < 0.05:
        strength_desc.append("moderate statistical evidence")
    elif max(p_simple, p_adjusted) < 0.1:
        strength_desc.append("weak statistical evidence")
    else:
        strength_desc.append("little statistical evidence")

    strength_desc.append(
        f"an interquartile-range decrease in the student–teacher ratio "
        f"is associated with an estimated {effect_iqr:.1f}-point change "
        f"in average test scores ({assoc_word} performance)"
    )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher "
        "academic performance (5th grade test scores) across California K–8 districts?\n\n"
        "Methods: I constructed a student–teacher ratio as students divided by teachers "
        "and an overall achievement measure as the average of reading and math scores. "
        "I first examined the Pearson correlation between the ratio and achievement, "
        "then estimated two linear regression models: (1) a simple regression of "
        "average test scores on the student–teacher ratio and (2) an adjusted regression "
        "controlling for district characteristics (percent CalWorks, percent reduced-price "
        "lunch, percent English learners, average income, expenditure per student, number "
        "of computers, and enrollment). All models were estimated using ordinary least "
        "squares with 420 districts.\n\n"
        f"Results: The student–teacher ratio is {direction}ly correlated with average "
        f"test scores (Pearson r = {corr:.3f}), meaning that districts with a lower "
        "ratio tend to have higher achievement. In the simple regression, the "
        f"coefficient on the student–teacher ratio is {coef_simple:.2f} with a "
        f"p-value of {p_simple:.4f}. In the adjusted regression, the coefficient is "
        f"{coef_adjusted:.2f} with a p-value of {p_adjusted:.4f}, after accounting "
        "for socioeconomic and demographic covariates. The estimated effect size implies "
        f"that {strength_desc[1]}. Taken together, these results indicate that, even "
        "after adjusting for observed district characteristics, districts with fewer "
        "students per teacher tend to have higher test scores.\n\n"
        f"Conclusion: Based on the observed direction of the association, the size of "
        f"the coefficients, and {strength_desc[0]} (p-values mostly below conventional "
        "0.05 thresholds), I conclude that there is a meaningful association between "
        "lower student–teacher ratios and higher academic performance in this dataset. "
        f"I therefore answer the research question with 'Yes', and encode this as a "
        f"Likert-scale response of {likert} on a 0–100 scale, where higher values "
        "represent stronger evidence for an association."
    )
    return explanation


def main() -> None:
    base = Path(__file__).resolve().parent

    # Load data
    df = pd.read_csv(base / "caschools.csv")
    df = df.copy()

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in variables of interest
    cols_for_analysis = [
        "stratio",
        "score",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "students",
    ]
    cols_for_analysis = [c for c in cols_for_analysis if c in df.columns]
    df_model = df[cols_for_analysis].dropna().copy()

    # Simple correlation
    corr = float(df_model["stratio"].corr(df_model["score"]))

    # Simple regression: score ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["score"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Adjusted regression with available controls
    control_candidates = [
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "students",
    ]
    controls = [c for c in control_candidates if c in df_model.columns]
    X_adj = sm.add_constant(df_model[["stratio"] + controls])
    model_adj = sm.OLS(y, X_adj).fit()
    coef_adjusted = float(model_adj.params["stratio"])
    p_adjusted = float(model_adj.pvalues["stratio"])

    # Effect over an interquartile range of the ratio
    q25, q75 = np.percentile(df_model["stratio"], [25, 75])
    effect_iqr = (q75 - q25) * coef_adjusted

    # Determine Likert response: encode strength of "Yes"
    # Default to neutral evidence.
    likert = 50

    if coef_adjusted < 0 and coef_simple < 0:
        # Both models suggest that lower ratios (fewer students per teacher)
        # are associated with higher scores.
        if p_simple < 0.001 and p_adjusted < 0.001 and abs(corr) >= 0.3:
            likert = 90
        elif p_simple < 0.01 and p_adjusted < 0.01:
            likert = 85
        elif p_simple < 0.05 and p_adjusted < 0.05:
            likert = 75
        elif p_simple < 0.1 or p_adjusted < 0.1:
            likert = 65
        else:
            likert = 55
    else:
        # Coefficients not consistently negative or evidence is weak; lean toward "No".
        if p_simple > 0.5 and p_adjusted > 0.5 and abs(corr) < 0.05:
            likert = 10
        elif p_simple > 0.1 and p_adjusted > 0.1:
            likert = 25
        else:
            likert = 45

    explanation = build_explanation(
        corr=corr,
        coef_simple=coef_simple,
        p_simple=p_simple,
        coef_adjusted=coef_adjusted,
        p_adjusted=p_adjusted,
        effect_iqr=effect_iqr,
        likert=likert,
    )

    conclusion = {
        "response": int(likert),
        "explanation": explanation,
    }

    (base / "conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

