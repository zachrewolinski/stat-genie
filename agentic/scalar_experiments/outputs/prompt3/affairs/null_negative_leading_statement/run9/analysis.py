import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Group-level descriptive statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_affair=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Logistic regression controlling for key covariates
    # Children is treated as a categorical variable with "no" as reference.
    model = smf.logit(
        "has_affair ~ C(children, Treatment(reference='no')) + age + yearsmarried + "
        "religiousness + education + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    coef_name = "C(children, Treatment(reference='no'))[T.yes]"
    coef_child_yes = float(model.params.get(coef_name, np.nan))
    pvalue_child_yes = float(model.pvalues.get(coef_name, np.nan))
    odds_ratio_child_yes = float(np.exp(coef_child_yes)) if np.isfinite(coef_child_yes) else np.nan

    # Determine direction: odds_ratio_child_yes < 1 implies children associated with fewer affairs.
    # We answer the question: "Does having children decrease engagement in extramarital affairs?"
    # If evidence points clearly to a decrease (odds_ratio < 1 with reasonably low p-value),
    # response is "Yes"; otherwise "No".
    if np.isfinite(odds_ratio_child_yes) and odds_ratio_child_yes < 1 and pvalue_child_yes < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Map statistical evidence to strength and confidence scores
    # Strength reflects magnitude of effect and clarity of descriptive differences.
    # Confidence reflects sample size, model fit, and p-value.
    # Start from a neutral baseline and adjust based on p-value and effect size.
    strength = 50
    confidence = 50

    if np.isfinite(odds_ratio_child_yes):
        # Effect size contribution
        effect_magnitude = abs(np.log(odds_ratio_child_yes))
        if effect_magnitude < 0.1:
            strength_adjust = -10
        elif effect_magnitude < 0.3:
            strength_adjust = 0
        elif effect_magnitude < 0.6:
            strength_adjust = 10
        else:
            strength_adjust = 20
        strength += strength_adjust

    if np.isfinite(pvalue_child_yes):
        if pvalue_child_yes < 0.001:
            confidence_adjust = 30
        elif pvalue_child_yes < 0.01:
            confidence_adjust = 20
        elif pvalue_child_yes < 0.05:
            confidence_adjust = 10
        elif pvalue_child_yes < 0.1:
            confidence_adjust = 0
        else:
            confidence_adjust = -10
        confidence += confidence_adjust

    # Clamp scores to [0, 100]
    strength = int(max(0, min(100, strength)))
    confidence = int(max(0, min(100, confidence)))

    # Build explanation text using both descriptive and model-based evidence
    children_yes_stats = group_stats.loc[group_stats["children"] == "yes"].iloc[0]
    children_no_stats = group_stats.loc[group_stats["children"] == "no"].iloc[0]

    explanation_parts = []
    explanation_parts.append(
        "The analysis examined whether having children is associated with decreased engagement "
        "in extramarital affairs, using the provided survey of 601 married individuals."
    )
    explanation_parts.append(
        f"Descriptively, the proportion reporting any extramarital affair was "
        f"{children_yes_stats['prop_affair']:.3f} among those with children (n={int(children_yes_stats['n'])}) "
        f"and {children_no_stats['prop_affair']:.3f} among those without children (n={int(children_no_stats['n'])}), "
        f"with mean affair scores of {children_yes_stats['mean_affairs']:.3f} and "
        f"{children_no_stats['mean_affairs']:.3f}, respectively."
    )

    if np.isfinite(odds_ratio_child_yes):
        explanation_parts.append(
            "A logistic regression model predicting the occurrence of any affair from children status, "
            "while controlling for age, years married, religiousness, education, marital rating, and gender, "
            f"estimated an odds ratio of {odds_ratio_child_yes:.3f} for respondents with children versus those without "
            f"(p-value = {pvalue_child_yes:.3g})."
        )
    else:
        explanation_parts.append(
            "A logistic regression model including children status and key covariates did not yield a stable "
            "estimate for the effect of children, suggesting the association is weak or highly uncertain."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the descriptive differences and the regression results indicate that having children is "
            "associated with a lower likelihood of engaging in extramarital affairs, supporting a 'Yes' answer "
            "to the research question."
        )
    else:
        explanation_parts.append(
            "Taken together, the descriptive statistics and regression results do not provide clear and robust "
            "evidence that having children decreases engagement in extramarital affairs; thus the answer to the "
            "research question is 'No'."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

