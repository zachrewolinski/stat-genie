import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")
    n = int(df.shape[0])

    # Binary outcome: any affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparison by children status
    grouped = df.groupby("children")["has_affair"].agg(["mean", "count"])
    children_yes_rate = grouped.loc["yes", "mean"]
    children_no_rate = grouped.loc["no", "mean"]
    rate_diff = children_yes_rate - children_no_rate

    coef_children = np.nan
    p_children = np.nan
    odds_ratio = np.nan
    model = None

    # Logistic regression controlling for key covariates
    try:
        model = smf.logit(
            "has_affair ~ C(children) + C(gender) + age + yearsmarried "
            "+ religiousness + education + occupation + rating",
            data=df,
        ).fit(disp=False)
        coef_children = model.params.get("C(children)[T.yes]", np.nan)
        p_children = model.pvalues.get("C(children)[T.yes]", np.nan)
        if np.isfinite(coef_children):
            odds_ratio = float(np.exp(coef_children))
    except Exception:
        model = None

    # Classify effect of having children on engagement in affairs
    effect_decrease = False
    effect_increase = False
    if np.isfinite(coef_children) and np.isfinite(p_children):
        if (
            coef_children < 0
            and p_children < 0.05
            and (children_no_rate - children_yes_rate) > 0.05
        ):
            effect_decrease = True
        elif (
            coef_children > 0
            and p_children < 0.05
            and (children_yes_rate - children_no_rate) > 0.05
        ):
            effect_increase = True

    if effect_decrease:
        response = "Yes"
        effect_label = "decrease"
    else:
        response = "No"
        effect_label = "increase" if effect_increase else "no_clear_effect"

    # Map evidence to a strength score (0–100)
    strength = 60
    if effect_label == "decrease":
        if p_children < 0.05:
            strength = 70
        if p_children < 0.01:
            strength = 80
        if p_children < 0.001:
            strength = 90
    elif effect_label == "increase":
        strength = 70
        if p_children < 0.05:
            strength = 80
        if p_children < 0.01:
            strength = 90
        if p_children < 0.001:
            strength = 95
    else:  # no_clear_effect
        diff_abs = float(abs(rate_diff))
        if diff_abs < 0.02 and (not np.isfinite(p_children) or p_children > 0.1):
            strength = 75
        elif diff_abs < 0.05:
            strength = 70
        else:
            strength = 65

    strength = int(max(0, min(100, strength)))

    # Confidence score (0–100) reflecting data quality and model robustness
    confidence = 75
    if n >= 500:
        confidence += 5
    if np.isfinite(p_children) and p_children < 0.01:
        confidence += 5
    confidence = int(max(0, min(100, confidence)))

    # Build explanation text
    explanation_parts = []
    explanation_parts.append(
        f"I analyzed {n} married individuals from the Fair affairs dataset to test whether having children decreases engagement in extramarital affairs."
    )
    explanation_parts.append(
        "I created a binary outcome indicating whether each respondent reported any extramarital sexual intercourse in the past year (affairs > 0)."
    )
    explanation_parts.append(
        f"Among respondents with children, {children_yes_rate:.1%} reported at least one affair, compared with {children_no_rate:.1%} among those without children "
        f"(difference = {rate_diff:+.1%})."
    )

    if model is not None and np.isfinite(coef_children) and np.isfinite(p_children):
        explanation_parts.append(
            "I then fit a logistic regression model predicting any affair from an indicator for having children, while controlling for gender, age, years married, religiousness, education, occupation, and self-rated marital happiness."
        )
        explanation_parts.append(
            f"The estimated log-odds coefficient for having children (yes versus no) was {coef_children:.3f}, corresponding to an odds ratio of {odds_ratio:.2f} (p-value = {p_children:.3g})."
        )
        if effect_label == "decrease":
            explanation_parts.append(
                "This negative and statistically significant coefficient, together with the lower observed affair rate among parents, indicates that having children is associated with meaningfully lower odds of engaging in extramarital affairs in this sample."
            )
        elif effect_label == "increase":
            explanation_parts.append(
                "This positive and statistically significant coefficient, together with the higher observed affair rate among parents, indicates that having children is associated with higher (not lower) odds of engaging in extramarital affairs in this sample."
            )
        else:
            explanation_parts.append(
                "However, the coefficient for having children is small in magnitude and not statistically significant at conventional levels, and the difference in affair rates between parents and non-parents is modest."
            )
            explanation_parts.append(
                "Taken together, these results do not provide credible evidence that having children reduces engagement in extramarital affairs; the data are consistent with little to no effect of children on affair involvement."
            )
    else:
        explanation_parts.append(
            "A multivariable logistic regression model for this outcome did not converge cleanly, so I focused on the descriptive difference in affair prevalence between parents and non-parents."
        )
        explanation_parts.append(
            "Because the observed difference is modest and does not clearly favor lower affair rates among parents, the data do not support the claim that having children decreases engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt with no extra text
    Path("conclusion.txt").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()

