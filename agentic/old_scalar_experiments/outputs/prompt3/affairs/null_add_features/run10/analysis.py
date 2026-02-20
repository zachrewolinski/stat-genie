import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary indicator: 1 = yes, 0 = no
    df["children_yes"] = df["children"].map({"yes": 1, "no": 0})

    # Basic group statistics for descriptive evidence
    group_stats = (
        df.groupby("children_yes")["affairs"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )

    # Logistic regression: probability of having at least one affair
    # Include key demographic and relationship covariates as controls
    predictors = [
        "children_yes",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    # One-hot encode gender while avoiding dummy variable trap
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=["gender"], drop_first=True)

    # Update predictor list to use encoded gender column if present
    encoded_gender_cols = [c for c in df_model.columns if c.startswith("gender_")]
    predictors_model = ["children_yes"] + encoded_gender_cols + [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    # Drop rows with missing predictors or outcome (should be none, but safe)
    model_data = df_model[predictors_model + ["has_affair"]].dropna()

    y = model_data["has_affair"]
    X = model_data[predictors_model]
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Extract effect of children on log-odds of having an affair
    coef_children = result.params["children_yes"]
    p_children = result.pvalues["children_yes"]
    or_children = float(np.exp(coef_children))

    # Descriptive effect: difference in mean number of affairs
    mean_with_children = float(group_stats.loc["has_children", "mean"])
    mean_without_children = float(group_stats.loc["no_children", "mean"])
    delta_mean = mean_without_children - mean_with_children

    # Determine overall direction of effect
    # Negative logistic coefficient and lower mean among parents both support
    # the claim that having children decreases engagement in affairs.
    supports_decrease = (coef_children < 0) and (delta_mean > 0)

    # Map statistical evidence to a binary Yes/No answer
    if supports_decrease:
        response = "Yes"
    else:
        response = "No"

    # Strength of the effect (0–100), based mainly on odds ratio magnitude
    # and consistency with mean difference.
    strength = compute_strength(or_children, delta_mean, supports_decrease)

    # Confidence in the conclusion (0–100), driven mostly by p-value and sample size
    confidence = compute_confidence(p_children, len(model_data), supports_decrease)

    explanation = build_explanation(
        response=response,
        coef_children=coef_children,
        or_children=or_children,
        p_children=p_children,
        mean_with_children=mean_with_children,
        mean_without_children=mean_without_children,
        delta_mean=delta_mean,
        strength=strength,
        confidence=confidence,
    )

    conclusion = {
        "response": response,
        "strength": int(round(strength)),
        "confidence": int(round(confidence)),
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt with no extra text
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


def compute_strength(
    odds_ratio: float, delta_mean: float, supports_decrease: bool
) -> float:
    """
    Map effect magnitude and consistency to a 0–100 strength score.
    """
    # Base strength from odds ratio alone
    if odds_ratio <= 0.5:
        base = 90.0
    elif odds_ratio <= 0.8:
        base = 75.0
    elif odds_ratio <= 0.95:
        base = 60.0
    elif odds_ratio < 1.05:
        base = 35.0
    else:
        base = 20.0

    # Adjust based on mean difference consistency
    if supports_decrease and delta_mean > 0:
        base += 5.0
    elif not supports_decrease and delta_mean <= 0:
        base += 5.0
    else:
        base -= 5.0

    return float(np.clip(base, 0.0, 100.0))


def compute_confidence(
    p_value: float, n_obs: int, supports_decrease: bool
) -> float:
    """
    Map p-value and sample size to a 0–100 confidence score.
    """
    # Start from p-value based tiers
    if p_value < 0.001:
        base = 90.0
    elif p_value < 0.01:
        base = 80.0
    elif p_value < 0.05:
        base = 70.0
    elif p_value < 0.1:
        base = 55.0
    else:
        base = 40.0

    # Sample size adjustment (dataset is moderately sized)
    if n_obs >= 500:
        base += 5.0
    elif n_obs >= 300:
        base += 2.0

    # Penalize if direction is not aligned across metrics
    if not supports_decrease:
        base -= 5.0

    return float(np.clip(base, 0.0, 100.0))


def build_explanation(
    response: str,
    coef_children: float,
    or_children: float,
    p_children: float,
    mean_with_children: float,
    mean_without_children: float,
    delta_mean: float,
    strength: float,
    confidence: float,
) -> str:
    """
    Construct a concise natural-language explanation summarizing the evidence.
    """
    direction_text = (
        "lower"
        if mean_with_children < mean_without_children
        else "higher"
        if mean_with_children > mean_without_children
        else "similar"
    )

    parts = []
    parts.append(
        "I fitted a logistic regression predicting whether a person had any "
        "extramarital affair in the past year from the presence of children in "
        "the marriage, controlling for gender, age, years married, religiousness, "
        "education, occupation, and marital satisfaction rating."
    )
    parts.append(
        f"The estimated coefficient for having children was {coef_children:.3f} "
        f"(odds ratio ≈ {or_children:.2f}, p-value ≈ {p_children:.3f})."
    )
    parts.append(
        f"On average, those with children had {mean_with_children:.2f} affairs per "
        f"year, while those without children had {mean_without_children:.2f}, a "
        f"difference of {delta_mean:.2f} (positive values mean fewer affairs among "
        "parents)."
    )
    parts.append(
        f"Based on this evidence, my answer to whether having children decreases "
        f"engagement in extramarital affairs is '{response}'. "
        f"The effect strength is summarized as {strength:.0f} on a 0–100 scale, "
        f"and my overall confidence in this conclusion is {confidence:.0f} on a "
        "0–100 scale."
    )

    return " ".join(parts)


if __name__ == "__main__":
    main()
