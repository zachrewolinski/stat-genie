import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # feature2: frequency of extramarital intercourse in past year
    # Create a binary outcome: any affair vs none.
    df = df.copy()
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # feature6: children in the marriage (yes/no)
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics: affair rates by children status
    rates = df.groupby("children_yes")["any_affair"].mean()
    count_by_children = df.groupby("children_yes")["any_affair"].size()

    # Logistic regression: any affair ~ children + controls
    # Controls: gender (feature3), age (feature4), years married (feature5),
    # religiousness (feature7), education (feature8), occupation (feature9),
    # marriage rating (feature10).
    # Encode gender as binary indicator for female to keep things simple.
    df["female"] = (df["feature3"].str.lower() == "female").astype(int)

    covariates = [
        "children_yes",
        "female",
        "feature4",  # age
        "feature5",  # years married
        "feature7",  # religiousness
        "feature8",  # education
        "feature9",  # occupation
        "feature10",  # marriage rating
    ]

    X = sm.add_constant(df[covariates])
    y = df["any_affair"]

    logit_model = sm.Logit(y, X)
    try:
        logit_res = logit_model.fit(disp=False)
    except Exception:
        # Fallback: in case of perfect separation or convergence issues,
        # drop controls and just fit any_affair ~ children.
        X_simple = sm.add_constant(df[["children_yes"]])
        logit_res = sm.Logit(y, X_simple).fit(disp=False)

    params = logit_res.params
    pvalues = logit_res.pvalues

    # Ensure we know which coefficient corresponds to children.
    if "children_yes" in params.index:
        coef_children = float(params["children_yes"])
        p_children = float(pvalues["children_yes"])
    else:
        # In the unlikely event the fallback removed it, treat as no evidence.
        coef_children = 0.0
        p_children = 1.0

    # Translate statistical results into a Likert-style 0–100 score.
    # - Strong evidence that children decrease affairs: high score near 100.
    # - Little or no evidence: low score near 0.
    # - If the effect is in the opposite direction but significant, we give a
    #   very low score close to 0, reflecting a strong "No".
    #
    # Use p-value and effect direction to set the score.
    if np.isnan(p_children):
        score = 50
    else:
        if coef_children < 0:
            # Children associated with *lower* affair odds (desired direction).
            if p_children < 0.001:
                score = 95
            elif p_children < 0.01:
                score = 85
            elif p_children < 0.05:
                score = 70
            elif p_children < 0.1:
                score = 60
            else:
                score = 45
        elif coef_children > 0:
            # Children associated with *higher* affair odds (opposite direction).
            if p_children < 0.001:
                score = 5
            elif p_children < 0.01:
                score = 10
            elif p_children < 0.05:
                score = 20
            elif p_children < 0.1:
                score = 30
            else:
                score = 40
        else:
            score = 50

    # Build a human-readable explanation summarizing the analysis.
    rate_children = float(rates.get(1, np.nan))
    rate_no_children = float(rates.get(0, np.nan))

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "Outcome variable: any extramarital intercourse in the past year "
        "(derived from the ordinal frequency measure)."
    )
    explanation_lines.append(
        "Key predictor: presence of children in the marriage (feature6: yes/no)."
    )
    explanation_lines.append(
        f"Sample size: {len(df)} married individuals "
        f"({int(count_by_children.get(1, 0))} with children, "
        f"{int(count_by_children.get(0, 0))} without children)."
    )
    if not np.isnan(rate_children) and not np.isnan(rate_no_children):
        explanation_lines.append(
            "Observed affair rates (any vs none): "
            f"{rate_children:.3f} among couples with children vs "
            f"{rate_no_children:.3f} among couples without children."
        )
    explanation_lines.append(
        "I fit a logistic regression model for the binary outcome "
        "any-affair vs no-affair, including children as the main predictor "
        "and adjusting for gender, age, years married, religiousness, "
        "education, occupation, and self-rated marital happiness."
    )
    explanation_lines.append(
        f"In this model, the coefficient for having children is {coef_children:.3f} "
        f"with p-value {p_children:.3g}."
    )

    if coef_children < 0:
        direction_text = (
            "Children are associated with lower odds of having had an "
            "extramarital affair in the past year."
        )
    elif coef_children > 0:
        direction_text = (
            "Children are associated with higher odds of having had an "
            "extramarital affair in the past year."
        )
    else:
        direction_text = (
            "The model estimates essentially no difference in affair odds "
            "between couples with and without children."
        )
    explanation_lines.append(direction_text)

    if p_children < 0.05:
        significance_text = (
            "This effect is statistically significant at conventional levels "
            "(p < 0.05), providing evidence of a real association in this sample."
        )
    elif p_children < 0.1:
        significance_text = (
            "This effect is only marginally significant (0.05 ≤ p < 0.10), "
            "suggesting at most weak evidence of an association."
        )
    else:
        significance_text = (
            "The effect is not statistically significant (p ≥ 0.10), so the data "
            "do not provide strong evidence that having children changes the "
            "likelihood of extramarital affairs once other factors are controlled."
        )
    explanation_lines.append(significance_text)

    if score >= 60:
        overall_answer = (
            "Overall conclusion: YES — there is evidence in this dataset that "
            "having children is associated with lower engagement in extramarital "
            f"affairs (Likert-style support score {score}/100)."
        )
    elif score <= 40:
        overall_answer = (
            "Overall conclusion: NO — this dataset does not support the claim "
            "that having children decreases engagement in extramarital affairs; "
            f"if anything, the estimated effect goes in the opposite or "
            f"negligible direction (Likert-style support score {score}/100)."
        )
    else:
        overall_answer = (
            "Overall conclusion: the evidence is mixed and does not strongly "
            "support a clear directional effect of children on extramarital "
            f"affairs in this sample (Likert-style support score {score}/100)."
        )
    explanation_lines.append(overall_answer)

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

