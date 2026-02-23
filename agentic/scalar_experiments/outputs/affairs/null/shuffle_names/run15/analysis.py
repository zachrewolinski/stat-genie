import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def compute_effect_and_pvalues(df: pd.DataFrame):
    """
    Compute group statistics, risk difference, and p-values for the relationship
    between having children and engagement in extramarital affairs.

    Returns a dict with:
      - n_with_children, n_without_children
      - prop_affair_with_children, prop_affair_without_children
      - mean_freq_with_children, mean_freq_without_children
      - risk_diff
      - z_pvalue
      - logit_odds_ratio
      - logit_pvalue
    """
    # Key variables
    df = df.copy()
    df["affair_freq"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    sub = df[["affair_freq", "any_affair", "has_children"]].dropna()

    # Group summaries
    group = sub.groupby("has_children")
    counts = group["any_affair"].sum()
    nobs = group["any_affair"].count()
    mean_any = counts / nobs
    mean_freq = group["affair_freq"].mean()

    # Ensure both groups (0 = no children, 1 = children) are present
    counts = counts.reindex([0, 1])
    nobs = nobs.reindex([0, 1])
    mean_any = mean_any.reindex([0, 1])
    mean_freq = mean_freq.reindex([0, 1])

    # Default values in case something is missing
    risk_diff = np.nan
    z_pvalue = np.nan
    logit_odds_ratio = np.nan
    logit_pvalue = np.nan

    # Difference in proportions test (any affair vs no affair)
    if counts.notna().all() and nobs.notna().all():
        try:
            # order: [no_children, children]
            z_stat, z_pvalue = proportions_ztest(count=counts.values, nobs=nobs.values)
            risk_diff = float(mean_any.loc[1] - mean_any.loc[0])
        except Exception:
            z_pvalue = np.nan

    # Logistic regression for any_affair ~ has_children
    try:
        X = sm.add_constant(sub["has_children"])
        y = sub["any_affair"]
        logit_model = sm.Logit(y, X).fit(disp=False)
        coef = logit_model.params.get("has_children", np.nan)
        logit_pvalue = float(logit_model.pvalues.get("has_children", np.nan))
        logit_odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan
    except Exception:
        logit_pvalue = np.nan
        logit_odds_ratio = np.nan

    return {
        "n_with_children": int(nobs.loc[1]) if not pd.isna(nobs.loc[1]) else 0,
        "n_without_children": int(nobs.loc[0]) if not pd.isna(nobs.loc[0]) else 0,
        "prop_affair_with_children": float(mean_any.loc[1]) if not pd.isna(mean_any.loc[1]) else np.nan,
        "prop_affair_without_children": float(mean_any.loc[0]) if not pd.isna(mean_any.loc[0]) else np.nan,
        "mean_freq_with_children": float(mean_freq.loc[1]) if not pd.isna(mean_freq.loc[1]) else np.nan,
        "mean_freq_without_children": float(mean_freq.loc[0]) if not pd.isna(mean_freq.loc[0]) else np.nan,
        "risk_diff": float(risk_diff) if not pd.isna(risk_diff) else np.nan,
        "z_pvalue": float(z_pvalue) if not pd.isna(z_pvalue) else np.nan,
        "logit_odds_ratio": float(logit_odds_ratio) if not pd.isna(logit_odds_ratio) else np.nan,
        "logit_pvalue": float(logit_pvalue) if not pd.isna(logit_pvalue) else np.nan,
    }


def map_to_likert(effect_stats: dict) -> int:
    """
    Map the statistical evidence to a 0–100 Likert scale
    answering: "Does having children decrease engagement in extramarital affairs?"

    0   = strong "No"
    100 = strong "Yes"
    """
    p = effect_stats["logit_pvalue"]
    if np.isnan(p):
        p = effect_stats["z_pvalue"]

    risk_diff = effect_stats["risk_diff"]

    # If we cannot compute anything reliable, return neutral-low "No"
    if np.isnan(risk_diff) or np.isnan(p):
        return 25

    # risk_diff = P(affair | children) - P(affair | no children)
    # Negative risk_diff => children associated with *fewer* affairs (supports the research question).
    if p <= 0.05:
        if risk_diff < 0:
            # Statistically significant decrease
            abs_diff = abs(risk_diff)
            if abs_diff > 0.15:
                return 90
            if abs_diff > 0.10:
                return 80
            if abs_diff > 0.05:
                return 75
            return 70
        else:
            # Statistically significant *increase* or no change
            abs_diff = abs(risk_diff)
            if abs_diff > 0.15:
                return 5
            if abs_diff > 0.10:
                return 10
            if abs_diff > 0.05:
                return 15
            return 20
    elif 0.05 < p <= 0.10:
        # Marginal evidence
        return 45 if risk_diff < 0 else 25
    else:
        # No statistically significant evidence of an effect
        return 35 if risk_diff < 0 else 20


def build_explanation(question: str, effect_stats: dict, response_score: int) -> str:
    """
    Build a concise, single-line textual explanation summarizing
    the statistical evidence and how it maps to the Likert score.
    """
    n_with = effect_stats["n_with_children"]
    n_without = effect_stats["n_without_children"]
    p_with = effect_stats["prop_affair_with_children"]
    p_without = effect_stats["prop_affair_without_children"]
    mean_freq_with = effect_stats["mean_freq_with_children"]
    mean_freq_without = effect_stats["mean_freq_without_children"]
    risk_diff = effect_stats["risk_diff"]
    z_p = effect_stats["z_pvalue"]
    logit_or = effect_stats["logit_odds_ratio"]
    logit_p = effect_stats["logit_pvalue"]

    parts = []
    parts.append(
        f"Research question: '{question.strip()}'"
    )
    parts.append(
        f"In the sample of {n_with + n_without} married individuals, {n_with} had children and {n_without} did not."
    )
    if not np.isnan(p_with) and not np.isnan(p_without):
        parts.append(
            f"The proportion engaging in any extramarital intercourse in the past year was "
            f"{p_with:.3f} among those with children and {p_without:.3f} among those without children "
            f"(risk difference children minus no children = {risk_diff:.3f})."
        )
    if not np.isnan(mean_freq_with) and not np.isnan(mean_freq_without):
        parts.append(
            f"Mean coded frequency of extramarital intercourse was {mean_freq_with:.3f} with children "
            f"versus {mean_freq_without:.3f} without children."
        )
    if not np.isnan(z_p):
        parts.append(
            f"A two-sample test for difference in proportions gave p-value {z_p:.3f}."
        )
    if not np.isnan(logit_or) and not np.isnan(logit_p):
        parts.append(
            f"A logistic regression of 'any affair' on 'having children' yielded an odds ratio of "
            f"{logit_or:.3f} for those with children relative to those without, with p-value {logit_p:.3f}."
        )

    if response_score >= 50:
        qualifier = "supports a 'Yes' answer that having children is associated with lower engagement in extramarital affairs"
    else:
        qualifier = "does not provide strong evidence that having children decreases engagement in extramarital affairs"

    parts.append(
        f"Overall, this pattern {qualifier}, leading to a Likert-scale response of {response_score} on a 0 (strong 'No') to 100 (strong 'Yes') scale."
    )

    # Single-line explanation (no newline characters) to comply strictly with instructions
    explanation = " ".join(parts)
    return explanation


def main():
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    with info_path.open("r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    effect_stats = compute_effect_and_pvalues(df)
    response_score = map_to_likert(effect_stats)
    explanation = build_explanation(question, effect_stats, response_score)

    result = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

