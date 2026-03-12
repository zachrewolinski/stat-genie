import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Define a binary outcome: any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    # Ensure children is treated as a categorical variable with explicit order
    df["children"] = df["children"].astype("category")
    df["gender"] = df["gender"].astype("category")
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    grouped = df.groupby("children")["has_affair"].agg(["mean", "count"])
    # Convert to plain Python types for serialization
    summary = {
        child: {
            "rate_any_affair": float(row["mean"]),
            "n": int(row["count"]),
        }
        for child, row in grouped.iterrows()
    }
    return summary


def fit_logistic_model(df: pd.DataFrame):
    # Logistic regression for having any affair, with children as key predictor
    # Include common controls to reduce confounding
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)
    return result


def extract_children_effect(result) -> dict:
    # With C(children), the baseline is typically the first category in sorted order.
    # We are interested in the contrast between children=yes and children=no.
    # The term name is usually like 'C(children)[T.yes]'.
    params = result.params
    pvalues = result.pvalues

    children_terms = {
        name: {
            "coef": float(params[name]),
            "pvalue": float(pvalues[name]),
        }
        for name in params.index
        if name.startswith("C(children)[")
    }

    # If no explicit children term exists (unexpected), fall back to zeros.
    if not children_terms:
        return {
            "term": None,
            "coef": 0.0,
            "pvalue": 1.0,
            "odds_ratio": 1.0,
        }

    # There should only be one contrast for children (yes vs no)
    term_name, stats = next(iter(children_terms.items()))
    coef = stats["coef"]
    pvalue = stats["pvalue"]
    odds_ratio = float(np.exp(coef))

    return {
        "term": term_name,
        "coef": coef,
        "pvalue": pvalue,
        "odds_ratio": odds_ratio,
    }


def map_to_likert(evidence: dict, group_summary: dict) -> int:
    """
    Map statistical evidence to a 0–100 Likert scale answering:
    'Does having children decrease engagement in extramarital affairs?'

    High values => strong 'Yes, children decrease affairs'.
    Low values  => strong 'No, children do not decrease affairs' (or may increase).
    """
    coef = evidence["coef"]
    pvalue = evidence["pvalue"]

    rate_children = group_summary.get("yes", {}).get("rate_any_affair", np.nan)
    rate_no_children = group_summary.get("no", {}).get("rate_any_affair", np.nan)

    # If rates are available, base direction on them; otherwise fall back to coef sign.
    if not np.isnan(rate_children) and not np.isnan(rate_no_children):
        diff = rate_children - rate_no_children  # negative if children associated with lower rate
    else:
        diff = None

    # Determine direction of effect with preference for observed rates
    if diff is not None and diff < 0:
        direction = "decrease"
    elif diff is not None and diff > 0:
        direction = "increase"
    else:
        # Tie or missing rates: use coefficient sign (OR < 1 => decrease)
        direction = "decrease" if coef < 0 else "increase_or_none"

    # Start from neutral
    score = 50

    # Adjust based on p-value (strength of evidence)
    if pvalue >= 0.1:
        strength_factor = 0.1
    elif pvalue >= 0.05:
        strength_factor = 0.3
    elif pvalue >= 0.01:
        strength_factor = 0.6
    else:
        strength_factor = 0.9

    # Adjust based on effect size in terms of absolute difference in rates if available
    effect_magnitude = 0.0
    if diff is not None:
        effect_magnitude = float(abs(diff))
    else:
        # As a fallback, use coefficient magnitude scaled down
        effect_magnitude = float(min(abs(coef) / 2.0, 0.3))

    # Cap effect magnitude to a reasonable range (since probabilities are between 0 and 1)
    effect_magnitude = max(0.0, min(effect_magnitude, 0.3))

    # Maximum possible shift away from neutral (50) based on combined strength
    max_shift = 50 * strength_factor * (effect_magnitude / 0.3 if effect_magnitude > 0 else 0.2)

    # For decreases (children reduce affairs), move upwards; otherwise, move downwards.
    if direction == "decrease":
        score = 50 + max_shift
    else:
        score = 50 - max_shift

    # Clamp to [0, 100] and cast to int
    score = int(round(max(0, min(100, score))))
    return score


def build_explanation(
    group_summary: dict,
    evidence: dict,
    likert_score: int,
    n: int,
) -> str:
    rate_children = group_summary.get("yes", {}).get("rate_any_affair", np.nan)
    n_children = group_summary.get("yes", {}).get("n", 0)
    rate_no_children = group_summary.get("no", {}).get("rate_any_affair", np.nan)
    n_no_children = group_summary.get("no", {}).get("n", 0)

    coef = evidence["coef"]
    pvalue = evidence["pvalue"]
    odds_ratio = evidence["odds_ratio"]

    # Qualitative summary of evidence direction
    if np.isnan(rate_children) or np.isnan(rate_no_children):
        direction_text = "The data did not clearly distinguish households with and without children."
    elif rate_children < rate_no_children:
        direction_text = (
            "In the raw data, respondents with children had a lower "
            "probability of any extramarital affair than those without children."
        )
    elif rate_children > rate_no_children:
        direction_text = (
            "In the raw data, respondents with children had a higher "
            "probability of any extramarital affair than those without children."
        )
    else:
        direction_text = (
            "In the raw data, respondents with and without children had very similar "
            "probabilities of any extramarital affair."
        )

    if pvalue < 0.05:
        sig_text = (
            "In a logistic regression controlling for age, years married, religiousness, "
            "education, occupation, marital satisfaction, and gender, the coefficient for "
            "having children was statistically significant (p < 0.05)."
        )
    elif pvalue < 0.1:
        sig_text = (
            "In the logistic regression with standard controls, the coefficient for "
            "having children was only marginally significant (0.05 ≤ p < 0.10)."
        )
    else:
        sig_text = (
            "In the logistic regression with standard controls, the coefficient for "
            "having children was not statistically significant (p ≥ 0.10), indicating "
            "limited evidence for a systematic effect."
        )

    if odds_ratio < 1:
        or_text = (
            f"The estimated odds ratio for having at least one affair when comparing "
            f"respondents with children to those without children was about "
            f"{odds_ratio:.2f} (logit coefficient {coef:.3f}, p-value {pvalue:.3f})."
        )
    else:
        or_text = (
            f"The estimated odds ratio for having at least one affair when comparing "
            f"respondents with children to those without children was about "
            f"{odds_ratio:.2f} (logit coefficient {coef:.3f}, p-value {pvalue:.3f}), "
            f"indicating equal or higher odds among those with children."
        )

    interpretation_parts = []
    if likert_score >= 60:
        interpretation_parts.append(
            "Overall, the data provide some support for the idea that having children "
            "is associated with a lower engagement in extramarital affairs."
        )
    elif likert_score <= 40:
        interpretation_parts.append(
            "Overall, the data do not support the claim that having children decreases "
            "engagement in extramarital affairs; if anything, the evidence points toward "
            "no clear reduction or even a slight increase."
        )
    else:
        interpretation_parts.append(
            "Overall, the data are broadly consistent with little to no systematic effect "
            "of having children on engagement in extramarital affairs."
        )

    interpretation_parts.append(
        "The Likert-scale response (0–100) reflects both the direction and strength of "
        "this evidence, where values near 0 indicate a strong 'No' to the hypothesis "
        "that children decrease affairs, values near 100 indicate a strong 'Yes', and "
        "values near 50 reflect equivocal or weak evidence."
    )

    explanation = (
        f"Sample and outcome:\n"
        f"- The dataset contains {n} married respondents.\n"
        f"- We defined a binary outcome 'has_affair' indicating whether the respondent "
        f"reported any extramarital sexual intercourse in the past year.\n\n"
        f"Descriptive comparison by children status:\n"
        f"- Among respondents with children (n = {n_children}), the proportion with any "
        f"extramarital affair was approximately {rate_children:.3f}.\n"
        f"- Among respondents without children (n = {n_no_children}), the proportion with any "
        f"extramarital affair was approximately {rate_no_children:.3f}.\n"
        f"- {direction_text}\n\n"
        f"Regression-based evidence:\n"
        f"- {sig_text}\n"
        f"- {or_text}\n\n"
        f"Conclusion and Likert-scale rating:\n"
        f"- Based on these analyses, the overall evidence regarding the hypothesis "
        f"that having children decreases engagement in extramarital affairs is "
        f"summarized by a Likert-scale response of {likert_score} on a scale from 0 "
        f"(strong 'No') to 100 (strong 'Yes').\n"
        f"- {' '.join(interpretation_parts)}"
    )

    return explanation


def main():
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    n = df.shape[0]
    group_summary = summarize_affairs_by_children(df)
    logit_result = fit_logistic_model(df)
    evidence = extract_children_effect(logit_result)

    likert_score = map_to_likert(evidence, group_summary)
    explanation = build_explanation(group_summary, evidence, likert_score, n)

    conclusion = {
        "response": int(likert_score),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

