import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Map columns to their semantic meaning using info.json descriptions
    # age column: frequency of extramarital intercourse in past year
    df["affairs_freq"] = df["age"]

    # religiousness column: actually indicates whether there are children in the marriage (yes/no)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Basic safety: drop rows with missing mappings, if any
    df = df.dropna(subset=["affairs_freq", "has_children"])

    # Create binary outcome: any affair in the past year
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)

    # Descriptive statistics by children status
    grouped = df.groupby("has_children")
    prevalence = grouped["any_affair"].mean()
    mean_freq = grouped["affairs_freq"].mean()
    counts = grouped["any_affair"].size()

    # Unadjusted logistic regression: any_affair ~ has_children
    y = df["any_affair"]
    X = sm.add_constant(df["has_children"])
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        coef_children = float(result.params["has_children"])
        p_value = float(result.pvalues["has_children"])
        intercept = float(result.params["const"])
    except Exception:
        # Fallback: no model results; treat as inconclusive
        coef_children = np.nan
        p_value = np.nan
        intercept = np.nan

    # Compute predicted probabilities from the logistic model if available
    def logistic(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    if not np.isnan(intercept) and not np.isnan(coef_children):
        prob_no_children = logistic(intercept)
        prob_children = logistic(intercept + coef_children)
        odds_ratio = float(np.exp(coef_children))
    else:
        prob_no_children = np.nan
        prob_children = np.nan
        odds_ratio = np.nan

    # Decide on the Likert-scale response (0–100, 0 = strong "No", 100 = strong "Yes")
    # Question: "Does having children decrease the engagement in extramarital affairs?"
    response_score: int
    direction = ""

    if not np.isnan(coef_children) and not np.isnan(p_value):
        if p_value < 0.05:
            if coef_children < 0:
                # Statistically significant evidence that having children is associated with fewer affairs
                response_score = 80
                direction = "decrease"
            else:
                # Statistically significant evidence in the opposite direction
                response_score = 20
                direction = "increase"
        else:
            # Coefficient not statistically different from zero
            response_score = 50
            direction = "no_clear_effect"
    else:
        # Model failed; rely on descriptive difference if available
        if 0 in prevalence.index and 1 in prevalence.index:
            diff = prevalence.loc[1] - prevalence.loc[0]
            if diff < 0:
                response_score = 60
                direction = "decrease_descriptive"
            elif diff > 0:
                response_score = 40
                direction = "increase_descriptive"
            else:
                response_score = 50
                direction = "no_clear_effect_descriptive"
        else:
            response_score = 50
            direction = "inconclusive"

    # Build explanation string with key numerical evidence
    parts = []

    total_n = int(len(df))
    parts.append(
        f"I analyzed {total_n} married individuals from the affairs dataset to examine whether having children is associated with less engagement in extramarital affairs."
    )

    if 0 in counts.index:
        n_no_children = int(counts.loc[0])
        prev_no_children = float(prevalence.loc[0]) * 100
        mean_aff_no_children = float(mean_freq.loc[0])
        parts.append(
            f"Among the {n_no_children} individuals without children, approximately {prev_no_children:.1f}% reported at least one extramarital intercourse in the past year, with an average affair-frequency score of {mean_aff_no_children:.2f}."
        )

    if 1 in counts.index:
        n_children = int(counts.loc[1])
        prev_children = float(prevalence.loc[1]) * 100
        mean_aff_children = float(mean_freq.loc[1])
        parts.append(
            f"Among the {n_children} individuals with children, approximately {prev_children:.1f}% reported at least one extramarital intercourse, with an average affair-frequency score of {mean_aff_children:.2f}."
        )

    if not np.isnan(coef_children):
        sign_word = "lower" if coef_children < 0 else "higher"
        parts.append(
            "I fit an unadjusted logistic regression model with a binary outcome indicating any affair in the past year and a predictor for whether there are children in the marriage."
        )
        if not np.isnan(odds_ratio) and not np.isnan(p_value):
            parts.append(
                f"In this model, having children was associated with {sign_word} odds of reporting an affair (odds ratio = {odds_ratio:.2f}, p-value = {p_value:.3f})."
            )

    if direction == "decrease":
        parts.append(
            "Because the children coefficient is negative and statistically significant, there is reasonably strong evidence that having children is associated with less engagement in extramarital affairs in this sample."
        )
    elif direction == "increase":
        parts.append(
            "Because the children coefficient is positive and statistically significant, the data suggest that having children is actually associated with more engagement in extramarital affairs in this sample rather than less."
        )
    elif direction == "no_clear_effect":
        parts.append(
            "Because the estimated effect of having children is small relative to its uncertainty (non-significant coefficient), the data do not provide clear evidence that having children meaningfully changes engagement in extramarital affairs."
        )
    elif direction.endswith("descriptive"):
        if "decrease" in direction:
            parts.append(
                "The regression model could not be reliably estimated, but descriptively the prevalence and average frequency of affairs are somewhat lower among individuals with children than among those without."
            )
        elif "increase" in direction:
            parts.append(
                "The regression model could not be reliably estimated, but descriptively the prevalence and average frequency of affairs are somewhat higher among individuals with children than among those without."
            )
        else:
            parts.append(
                "Both descriptive comparisons and the attempted regression model are inconclusive about any difference in affairs between individuals with and without children."
            )
    else:
        parts.append(
            "Overall, model estimation was unstable, so the evidence about whether children affect engagement in extramarital affairs is inconclusive in this dataset."
        )

    explanation = " ".join(parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write strictly the JSON object to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

