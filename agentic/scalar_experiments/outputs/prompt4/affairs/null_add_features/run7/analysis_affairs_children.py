import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Basic cleaning / derived variables
    df = df.dropna(subset=["affairs", "children"])
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = df["children"].astype(str).str.lower().eq("yes").astype(int)

    # Descriptive statistics by children status
    group = df.groupby("children")
    counts = group["affairs"].size().to_dict()
    mean_affairs = group["affairs"].mean().to_dict()
    median_affairs = group["affairs"].median().to_dict()
    prop_has_affair = group["has_affair"].mean().to_dict()

    # Difference in probability of any affair
    prop_children_yes = prop_has_affair.get("yes", np.nan)
    prop_children_no = prop_has_affair.get("no", np.nan)
    diff_prop = prop_children_no - prop_children_yes

    # Logistic regression: has_affair ~ children_yes (+ controls)
    # Basic model
    X_basic = sm.add_constant(df[["children_yes"]])
    y = df["has_affair"]
    coef_basic = np.nan
    p_basic = np.nan
    odds_basic = np.nan
    try:
        model_basic = sm.Logit(y, X_basic).fit(disp=False)
        coef_basic = float(model_basic.params["children_yes"])
        p_basic = float(model_basic.pvalues["children_yes"])
        odds_basic = float(np.exp(coef_basic))
    except Exception:
        pass

    # Model with standard controls from the classic affairs dataset
    coef_full = np.nan
    p_full = np.nan
    odds_full = np.nan
    controls = []
    for col in ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]:
        if col in df.columns:
            controls.append(col)

    if controls:
        X_full = sm.add_constant(df[["children_yes"] + controls])
        try:
            model_full = sm.Logit(y, X_full).fit(disp=False)
            coef_full = float(model_full.params["children_yes"])
            p_full = float(model_full.pvalues["children_yes"])
            odds_full = float(np.exp(coef_full))
        except Exception:
            pass

    # Map evidence to a 0–100 Likert score
    # Direction: positive if having children is associated with *lower* engagement.
    direction = 0.0
    if np.isfinite(diff_prop) and diff_prop != 0:
        direction = 1.0 if diff_prop > 0 else -1.0

    # Effect magnitude based on difference in proportions (cap at 0.20 as "strong")
    effect_size = float(abs(diff_prop)) if np.isfinite(diff_prop) else 0.0
    effect_magnitude_score = min(effect_size / 0.20, 1.0)

    # Significance factor from logistic regressions
    sig_evidence = 0.0
    negative_and_sig = []
    for coef, pval in [(coef_basic, p_basic), (coef_full, p_full)]:
        if np.isfinite(coef) and np.isfinite(pval) and pval < 0.05:
            negative_and_sig.append(coef < 0)

    if negative_and_sig:
        # At least one significant model; check whether they consistently support the same direction
        share_negative = sum(negative_and_sig) / len(negative_and_sig)
        # If most significant models are negative (children decrease affairs), boost evidence
        sig_evidence = 0.8 * share_negative + 0.2 * (1.0 - share_negative)
    else:
        # No significant models; downweight regression evidence
        sig_evidence = 0.4

    combined_strength = direction * effect_magnitude_score * sig_evidence
    response = int(round(50 + 50 * combined_strength))
    response = max(0, min(100, response))

    # Build a human-readable explanation
    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%" if np.isfinite(x) else "NA"

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease the engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"Sample size after dropping missing values: {len(df)} married individuals."
    )
    explanation_lines.append(
        f"Group sizes by children status: {counts}."
    )
    explanation_lines.append(
        "Mean number of affairs in the past year by children status "
        f"(0 = none, larger values = more frequent): "
        f"{ {k: round(v, 3) for k, v in mean_affairs.items()} }."
    )
    explanation_lines.append(
        "Median number of affairs by children status: "
        f"{ {k: float(v) for k, v in median_affairs.items()} }."
    )
    explanation_lines.append(
        "Proportion with at least one affair by children status: "
        f"{ {k: fmt_pct(v) for k, v in prop_has_affair.items()} }."
    )
    explanation_lines.append(
        f"Difference in probability of any affair (no children minus children): {diff_prop:.3f}."
    )

    if np.isfinite(coef_basic):
        explanation_lines.append(
            "Logistic regression (has_affair ~ children only): "
            f"children_yes coefficient = {coef_basic:.3f}, p = {p_basic:.3g}, "
            f"odds ratio = {odds_basic:.3f}."
        )
    else:
        explanation_lines.append(
            "Logistic regression with children only could not be fit reliably."
        )

    if np.isfinite(coef_full):
        explanation_lines.append(
            "Logistic regression with controls for age, years married, religiousness, "
            "education, occupation, and marital rating: "
            f"children_yes coefficient = {coef_full:.3f}, p = {p_full:.3g}, "
            f"odds ratio = {odds_full:.3f}."
        )
    else:
        explanation_lines.append(
            "A logistic regression with standard controls could not be fit reliably."
        )

    if direction > 0:
        qualitative = (
            "Overall, individuals with children appear somewhat less likely to engage in "
            "extramarital affairs than those without children, "
            "though the effect size and statistical strength are limited."
        )
    elif direction < 0:
        qualitative = (
            "Overall, individuals with children do not appear less likely to engage in "
            "extramarital affairs; if anything, the association runs in the opposite "
            "direction, and evidence for a protective effect of having children is weak."
        )
    else:
        qualitative = (
            "Overall, the data do not show a clear difference in engagement in "
            "extramarital affairs between individuals with and without children."
        )

    explanation_lines.append(qualitative)
    explanation_lines.append(
        "The numeric response on the 0–100 scale summarizes this evidence, where 0 means "
        "a strong 'No' to the claim that having children decreases affairs, 50 means no "
        "clear evidence either way, and 100 means a strong 'Yes'."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": response, "explanation": explanation}

    # Write JSON conclusion to conclusion.txt (only the JSON object, no extra text)
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

