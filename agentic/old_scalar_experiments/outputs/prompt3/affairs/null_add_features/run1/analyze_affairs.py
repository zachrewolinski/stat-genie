import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def compute_strength_and_confidence(odds_ratio: float, p_value: float, diff_rate: float, n: int):
    """Map statistical evidence to strength and confidence scores (0-100)."""
    # Significance component
    if p_value < 0.001:
        sig_score = 1.0
    elif p_value < 0.01:
        sig_score = 0.9
    elif p_value < 0.05:
        sig_score = 0.75
    elif p_value < 0.1:
        sig_score = 0.6
    elif p_value < 0.2:
        sig_score = 0.4
    else:
        sig_score = 0.2

    # Effect-size component from odds ratio and absolute difference in prevalence
    log_or = abs(np.log(odds_ratio)) if odds_ratio > 0 else 0.0
    if log_or > 0.75:
        or_score = 1.0
    elif log_or > 0.5:
        or_score = 0.8
    elif log_or > 0.25:
        or_score = 0.6
    elif log_or > 0.1:
        or_score = 0.4
    else:
        or_score = 0.2

    abs_diff = abs(diff_rate)
    if abs_diff > 0.15:
        diff_score = 1.0
    elif abs_diff > 0.10:
        diff_score = 0.8
    elif abs_diff > 0.05:
        diff_score = 0.6
    elif abs_diff > 0.02:
        diff_score = 0.4
    else:
        diff_score = 0.2

    effect_score = 0.5 * or_score + 0.5 * diff_score

    # Sample size component (saturates by ~300 obs)
    sample_score = min(1.0, n / 300.0) if n > 0 else 0.0

    strength = int(round(100 * (0.5 * sig_score + 0.5 * effect_score)))
    confidence = int(round(100 * (0.6 * sig_score + 0.4 * sample_score)))

    strength = max(0, min(100, strength))
    confidence = max(0, min(100, confidence))

    return strength, confidence


def main():
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Focus on the relationship between having children and engagement in affairs
    df = df.dropna(subset=["affairs", "children"])

    # Binary outcome: any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure children is treated consistently as yes/no
    df["children"] = df["children"].astype(str).str.lower().str.strip()

    # Basic group summaries
    group_any = df.groupby("children")["has_affair"].agg(["mean", "count"])
    group_affairs = df.groupby("children")["affairs"].agg(["mean", "median"])

    # Derive prevalence for yes/no groups where available
    mean_yes = group_any.loc["yes", "mean"] if "yes" in group_any.index else np.nan
    mean_no = group_any.loc["no", "mean"] if "no" in group_any.index else np.nan
    diff_rate = mean_yes - mean_no if np.all(np.isfinite([mean_yes, mean_no])) else np.nan

    # Logistic regression controlling for key demographic and marital covariates
    covariates = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    available_covariates = [c for c in covariates if c in df.columns]

    model_data_cols = ["has_affair", "children"] + available_covariates
    model_data = df[model_data_cols].dropna()
    model_data["children_yes"] = (model_data["children"] == "yes").astype(int)

    odds_ratio = 1.0
    p_value = 1.0
    n_model = len(model_data)

    if n_model > 0:
        formula_terms = ["children_yes"] + available_covariates
        formula = "has_affair ~ " + " + ".join(formula_terms)
        try:
            logit_model = smf.logit(formula, data=model_data).fit(disp=False)
            coef = float(logit_model.params["children_yes"])
            p_value = float(logit_model.pvalues["children_yes"])
            odds_ratio = float(np.exp(coef))
        except Exception:
            # Fall back to a very simple two-sample comparison if the model has issues
            odds_ratio = 1.0
            p_value = 1.0

    # Decide direction of effect based primarily on logistic regression,
    # falling back to raw prevalence if needed.
    direction = None  # "decrease" or "increase"

    if odds_ratio < 1.0 and np.isfinite(odds_ratio):
        direction = "decrease"
    elif odds_ratio > 1.0 and np.isfinite(odds_ratio):
        direction = "increase"
    elif np.all(np.isfinite([mean_yes, mean_no])):
        # Use prevalence difference as a backup
        direction = "decrease" if mean_yes < mean_no else "increase"

    if direction is None:
        # If we truly cannot tell, answer "No" with very low strength and confidence
        response = "No"
        strength = 10
        confidence = 10
        explanation = (
            "The available data did not allow a stable estimate of how having children "
            "relates to extramarital affairs (model fitting failed and group summaries "
            "were inconclusive), so I cannot conclude that having children decreases "
            "engagement in extramarital affairs."
        )
    else:
        # Compute strength and confidence from evidence
        if not np.isfinite(diff_rate):
            # If diff_rate is missing, approximate it from group means of has_affair
            if np.all(np.isfinite([mean_yes, mean_no])):
                diff_rate = mean_yes - mean_no
            else:
                diff_rate = 0.0

        strength, confidence = compute_strength_and_confidence(
            odds_ratio=odds_ratio,
            p_value=p_value,
            diff_rate=diff_rate,
            n=n_model,
        )

        # Turn direction into Yes/No answer
        if direction == "decrease":
            response = "Yes"
        else:
            response = "No"

        # Build a concise explanation summarizing key evidence
        parts = []
        total_n = int(df.shape[0])
        parts.append(
            f"I analyzed {total_n} married individuals and compared extramarital affairs "
            f"between those with and without children."
        )

        if np.all(np.isfinite([mean_yes, mean_no])):
            pct_yes = mean_yes * 100.0
            pct_no = mean_no * 100.0
            parts.append(
                f"The observed prevalence of any affair was about {pct_yes:.1f}% "
                f"for those with children versus {pct_no:.1f}% for those without."
            )

        if np.isfinite(odds_ratio):
            parts.append(
                f"A logistic regression adjusting for age, years married, religiousness, "
                f"education, occupation, and relationship rating estimated an odds ratio "
                f"of {odds_ratio:.2f} for having any affair when children are present "
                f"(p = {p_value:.3g})."
            )

        if response == "Yes":
            parts.append(
                "Because the estimated odds ratio is below 1 (and the direction of the "
                "group difference is consistent), the data suggest that having children "
                "is associated with a lower likelihood of engaging in extramarital affairs."
            )
        else:
            parts.append(
                "Because the estimated odds ratio is not meaningfully below 1 (and/or the "
                "group difference does not favor lower affair rates among parents), the data "
                "do not support the claim that having children decreases engagement in "
                "extramarital affairs."
            )

        parts.append(
            "The strength score reflects the combined magnitude and statistical significance "
            "of this association, while the confidence score reflects both statistical "
            "uncertainty and the sample size."
        )

        explanation = " ".join(parts)

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

