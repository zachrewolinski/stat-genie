import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affairs in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by presence of children
    prob_by_children = df.groupby("children")["affair_any"].mean()
    mean_affairs_by_children = df.groupby("children")["affairs"].mean()

    # Logistic regression controlling for key covariates
    model = smf.logit(
        "affair_any ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)

    # Coefficient for having children (yes vs no)
    coef_children = model.params.get("C(children)[T.yes]")
    pval_children = model.pvalues.get("C(children)[T.yes]")

    # Unadjusted probabilities
    prob_children_yes = float(prob_by_children.get("yes", float("nan")))
    prob_children_no = float(prob_by_children.get("no", float("nan")))

    # Decide direction of effect
    unadj_direction_children_less = prob_children_yes < prob_children_no
    model_direction_children_less = coef_children is not None and coef_children < 0

    # Heuristic mapping from evidence strength to 0–100 scale
    if unadj_direction_children_less and model_direction_children_less:
        # Both descriptive and model-based analyses indicate fewer affairs with children
        if pval_children is not None and pval_children < 0.01:
            response = "Yes"
            strength = 90
            confidence = 85
        elif pval_children is not None and pval_children < 0.05:
            response = "Yes"
            strength = 80
            confidence = 80
        elif pval_children is not None and pval_children < 0.1:
            response = "Yes"
            strength = 65
            confidence = 65
        else:
            response = "Yes"
            strength = 55
            confidence = 55
    elif (not unadj_direction_children_less) and (not model_direction_children_less):
        # Both analyses suggest equal or higher affair risk with children
        if pval_children is not None and pval_children < 0.01:
            response = "No"
            strength = 90
            confidence = 85
        elif pval_children is not None and pval_children < 0.05:
            response = "No"
            strength = 80
            confidence = 80
        elif pval_children is not None and pval_children < 0.1:
            response = "No"
            strength = 65
            confidence = 65
        else:
            response = "No"
            strength = 55
            confidence = 55
    else:
        # Descriptive and model-based results disagree; treat as weaker evidence
        response = "No"
        strength = 40
        confidence = 45

    # Build explanation text
    explanation_parts = []
    explanation_parts.append(
        "I analyzed the Psychology Today extramarital affairs dataset (601 married individuals) "
        "to test whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_parts.append(
        f"First, I created a binary indicator for having at least one affair in the past year and "
        f"compared this probability by child status: with children, the estimated probability was "
        f"{prob_children_yes:.3f}, versus {prob_children_no:.3f} without children."
    )
    explanation_parts.append(
        "I then fit a logistic regression model predicting the presence of any affair from child status, "
        "controlling for age, years married, religiousness, education, occupation, and self-rated marital happiness."
    )
    if coef_children is not None and pval_children is not None:
        explanation_parts.append(
            f"In this model, the coefficient for having children (yes vs no) was {coef_children:.3f} with "
            f"a p-value of {pval_children:.3f}, indicating "
            f"{'lower' if coef_children < 0 else 'higher' if coef_children > 0 else 'no clear change'} "
            f"log-odds of an affair for parents relative to non-parents after adjustment."
        )
    explanation_parts.append(
        "I combined the direction and statistical significance of both the unadjusted comparison and the adjusted "
        "regression effect to decide whether the evidence supports the claim that having children decreases "
        "engagement in extramarital affairs, and then mapped that evidence to strength and confidence scores "
        "on a 0–100 scale."
    )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()

