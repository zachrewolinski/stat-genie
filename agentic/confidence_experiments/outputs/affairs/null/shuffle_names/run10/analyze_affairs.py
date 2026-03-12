import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_likert_from_effect(p_value: float, effect: float) -> int:
    """
    Map statistical evidence to a 0-100 Likert score where
    higher values indicate stronger evidence that having children
    decreases engagement in extramarital affairs.

    Here, a positive `effect` means fewer affairs when children
    are present (i.e., the data favor a \"Yes\" answer).
    """
    magnitude = abs(effect)

    # Base movement away from 50 driven by p-value (evidence strength)
    if p_value >= 0.10:
        base_delta = 5
    elif p_value >= 0.05:
        base_delta = 15
    elif p_value >= 0.01:
        base_delta = 25
    else:
        base_delta = 35

    # Modulate by effect size
    if magnitude < 0.05:
        mag_factor = 0.3
    elif magnitude < 0.15:
        mag_factor = 0.6
    elif magnitude < 0.30:
        mag_factor = 0.8
    else:
        mag_factor = 1.0

    delta = base_delta * mag_factor
    if effect < 0:
        delta = -delta

    score = int(round(np.clip(50 + delta, 0, 100)))
    return score


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json metadata, `religiousness` encodes
    # "Are there children in the marriage?" with values yes/no.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # According to metadata, `age` encodes the affair frequency categories.
    df["affair_freq"] = df["age"]

    # Basic group stats
    grouped = df.groupby("has_children")["affair_freq"]
    means = grouped.mean()
    stds = grouped.std()

    # Linear model: affair frequency ~ has_children
    # (unadjusted, since the research question focuses on this relationship)
    model = smf.ols("affair_freq ~ has_children", data=df).fit()
    coef_children = model.params["has_children"]
    p_value = model.pvalues["has_children"]

    # In this parameterization, coef_children equals
    # mean(with children) - mean(without children).
    # For the research question, define effect_for_question so that
    # positive values support \"children decrease affairs\".
    effect_for_question = -coef_children

    likert_score = compute_likert_from_effect(
        p_value=p_value, effect=effect_for_question
    )

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "Dataset: 601 married individuals from the Psychology Today survey as summarized in Fair (1978)."
    )
    explanation_lines.append(
        "Per the provided metadata, the column 'religiousness' encodes whether there are children in the marriage "
        "('yes'/'no'), and the column 'age' actually encodes categories of extramarital sexual intercourse frequency."
    )
    explanation_lines.append(
        f"Sample sizes: {int(df['has_children'].sum())} with children and "
        f"{int((1 - df['has_children']).sum())} without children."
    )
    explanation_lines.append(
        f"Mean affair-frequency code (0,1,2,3,7,12): "
        f"{means[1]:.3f} for couples with children vs. {means[0]:.3f} for couples without children."
    )
    explanation_lines.append(
        f"Standard deviations: {stds[1]:.3f} (with children) vs. {stds[0]:.3f} (without children)."
    )
    explanation_lines.append(
        "I fit an ordinary least squares model affair_freq ~ has_children to test the mean difference."
    )
    explanation_lines.append(
        f"The estimated OLS coefficient on has_children (mean with children minus mean without children) "
        f"is {coef_children:.3f} with p-value {p_value:.4f}."
    )

    if p_value < 0.05 and coef_children > 0:
        direction_comment = (
            "The positive coefficient indicates that, on average, couples with children have *higher* "
            "affair-frequency codes than couples without children."
        )
        conclusion_comment = (
            "This provides evidence against the hypothesis that having children decreases engagement in extramarital affairs."
        )
    elif p_value < 0.05 and coef_children < 0:
        direction_comment = (
            "The negative coefficient indicates that, on average, couples with children have lower "
            "affair-frequency codes than couples without children."
        )
        conclusion_comment = (
            "This supports the hypothesis that having children is associated with fewer extramarital affairs, "
            "although the effect size should be interpreted in the context of the coded scale."
        )
    else:
        direction_comment = (
            "The estimated coefficient is not statistically distinguishable from zero at conventional levels."
        )
        conclusion_comment = (
            "Thus, the data do not provide strong evidence that having children meaningfully changes engagement "
            "in extramarital affairs."
        )

    explanation_lines.append(direction_comment)
    explanation_lines.append(conclusion_comment)
    explanation_lines.append(
        f"The Likert-scale response ({likert_score} on a 0–100 scale where higher values correspond to a stronger 'Yes' "
        "answer that children decrease affairs) reflects both the statistical significance and the magnitude of the "
        "estimated effect."
    )

    explanation = " ".join(explanation_lines)

    result = {"response": likert_score, "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
