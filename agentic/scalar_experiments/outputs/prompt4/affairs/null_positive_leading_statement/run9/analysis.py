import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_score(child_or: float, child_p: float, dp: float, dmean: float) -> int:
    """
    Map statistical evidence to a 0–100 score where
    0 = strong "No" and 100 = strong "Yes" to:
    "Does having children decrease engagement in extramarital affairs?"
    """
    # Base on direction and significance of the children effect in the logistic model
    if child_or < 1:
        if child_p < 0.01:
            score = 90
        elif child_p < 0.05:
            score = 80
        elif child_p < 0.1:
            score = 65
        else:
            score = 55
    elif child_or > 1:
        if child_p < 0.01:
            score = 10
        elif child_p < 0.05:
            score = 20
        elif child_p < 0.1:
            score = 35
        else:
            score = 45
    else:
        score = 50

    # Adjust using descriptive differences in probabilities (dp) and mean counts (dmean)
    # dp: P(affair | children=yes) - P(affair | children=no)
    if dp < -0.05:
        score += 5
    elif dp > 0.05:
        score -= 5

    # dmean: mean_affairs(children=yes) - mean_affairs(children=no)
    if dmean < -0.5:
        score += 5
    elif dmean > 0.5:
        score -= 5

    # Clip to [0, 100] and ensure integer
    score = int(round(score))
    score = max(0, min(100, score))
    return score


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Indicator for having children: 1 = yes, 0 = no
    df["child_indicator"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Descriptive statistics by children status
    group = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prob_any_affair=("any_affair", "mean"),
            count=("any_affair", "size"),
        )
        .sort_index()
    )

    # Extract descriptive metrics (ensure both "no" and "yes" exist)
    # If a level is missing, fall back gracefully to NaNs.
    def safe_get(level: str, col: str) -> float:
        if level in group.index:
            return float(group.loc[level, col])
        return float("nan")

    mean_aff_yes = safe_get("yes", "mean_affairs")
    mean_aff_no = safe_get("no", "mean_affairs")
    prob_any_yes = safe_get("yes", "prob_any_affair")
    prob_any_no = safe_get("no", "prob_any_affair")

    dmean = mean_aff_yes - mean_aff_no
    dp = prob_any_yes - prob_any_no

    # Logistic regression: any_affair ~ children + controls
    X = df[
        [
            "child_indicator",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "rating",
        ]
    ].copy()

    # Add categorical controls via one-hot encoding
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)
    occ_dummies = pd.get_dummies(df["occupation"].astype("category"), prefix="occ", drop_first=True)
    X = pd.concat([X, gender_dummies, occ_dummies], axis=1)

    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    child_coef = float(logit_model.params["child_indicator"])
    child_p = float(logit_model.pvalues["child_indicator"])
    child_or = float(np.exp(child_coef))

    # Compute 0–100 score summarizing evidence
    score = compute_score(child_or=child_or, child_p=child_p, dp=dp, dmean=dmean)

    # Qualitative summary based on the score and direction
    if score >= 70:
        qualitative = (
            "The data provide strong evidence that having children is "
            "associated with fewer extramarital affairs."
        )
    elif score >= 60:
        qualitative = (
            "The data provide moderate evidence that having children is "
            "associated with fewer extramarital affairs."
        )
    elif score <= 30:
        qualitative = (
            "The data provide strong evidence against the idea that having "
            "children reduces extramarital affairs, and instead suggest that "
            "parents may be at least as likely to have affairs as non-parents."
        )
    elif score <= 40:
        qualitative = (
            "The data provide modest evidence against the idea that having "
            "children reduces extramarital affairs."
        )
    else:
        qualitative = (
            "The data do not provide clear evidence that having children "
            "meaningfully changes engagement in extramarital affairs."
        )

    direction_desc = "decrease" if child_or < 1 else "increase"

    explanation_lines = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
        "Dataset: 601 currently married individuals from the Psychology Today survey as compiled in the Fair (1978) 'Affairs' dataset.",
        (
            "Descriptive results: mean affair count is "
            f"{mean_aff_yes:.3f} for individuals with children and "
            f"{mean_aff_no:.3f} for those without children; the proportion "
            f"with any affairs is {prob_any_yes:.3f} for individuals with "
            f"children versus {prob_any_no:.3f} for those without."
        ),
        (
            "Model-based results: a logistic regression of having any affair "
            "on an indicator for having children, controlling for age, years "
            "married, gender, occupation, religiousness, education, and "
            "self-rated marital happiness, yields an odds ratio of "
            f"{child_or:.3f} for individuals with children relative to those "
            f"without (log-odds coefficient {child_coef:.3f}, p-value "
            f"{child_p:.4f}). An odds ratio below 1 would indicate that "
            "having children is associated with fewer affairs, while an odds "
            "ratio above 1 would indicate more affairs."
        ),
        (
            f"In this analysis, the direction of the estimated effect suggests "
            f"that having children tends to {direction_desc} engagement in "
            "extramarital affairs. The strength and statistical significance "
            "of this effect, together with the descriptive differences in mean "
            "affair counts and the probability of any affair between parents "
            "and non-parents, inform the overall confidence score."
        ),
        (
            f"Summary interpretation: {qualitative} On a 0–100 scale where 0 "
            f"represents a strong 'No' and 100 represents a strong 'Yes' to "
            f"the question 'Does having children decrease engagement in "
            f"extramarital affairs?', the evidence from this dataset "
            f"corresponds to a score of {score}."
        ),
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": score, "explanation": explanation}

    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

