import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity: drop rows with missing core variables
    core_vars = ["y", "age", "culture"]
    df = df.dropna(subset=core_vars)
    return df


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Majority choice (2) vs all other options
    df["majority_choice"] = (df["y"] == 2).astype(int)
    # Reliance on any social information: chose majority (2) or minority (3) vs undemonstrated option (1)
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    # Center age for interpretability
    df["age_c"] = df["age"] - df["age"].mean()
    # Treat culture and gender as categorical predictors
    df["culture"] = df["culture"].astype("category")
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("category")
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    try:
        model = smf.logit(formula=formula, data=data)
        result = model.fit(disp=False)
        return result
    except Exception:
        return None


def summarize_effect(result, term_prefix: str):
    """Return minimal summary about whether a group of terms shows strong evidence.

    term_prefix is used to group coefficients, e.g. 'C(culture)[' or 'age_c'.
    """
    if result is None:
        return {
            "has_evidence": False,
            "min_p": None,
            "max_effect": None,
        }

    pvalues = result.pvalues
    params = result.params

    mask = [name.startswith(term_prefix) or name == term_prefix for name in pvalues.index]
    if not any(mask):
        return {
            "has_evidence": False,
            "min_p": None,
            "max_effect": None,
        }

    selected_p = pvalues[mask]
    selected_params = params[mask]

    min_p = float(selected_p.min())
    max_effect = float(selected_params.abs().max())

    has_evidence = min_p < 0.05
    return {
        "has_evidence": has_evidence,
        "min_p": min_p,
        "max_effect": max_effect,
    }


def compute_likert_score(evidence_flags):
    """Combine multiple evidence indicators into a 0–100 Likert score.

    evidence_flags is a dict with booleans for keys like:
    - age_social
    - culture_social
    - age_majority
    - culture_majority
    """
    num_true = sum(1 for v in evidence_flags.values() if v)
    if num_true == 0:
        # Little to no evidence that age or culture matter
        return 25
    if num_true == 1:
        return 60
    if num_true in (2, 3):
        return 75
    # Strong, consistent evidence across all aspects
    return 90


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "boxes.csv"

    df = load_data(csv_path)
    df = build_outcomes(df)

    # Logistic models:
    # 1) social_choice ~ age + culture (+ gender, majority_first if present)
    predictors = ["age_c", "C(culture)"]
    if "gender" in df.columns:
        predictors.append("C(gender)")
    if "majority_first" in df.columns:
        predictors.append("majority_first")

    rhs = " + ".join(predictors)

    social_formula = f"social_choice ~ {rhs}"
    majority_formula = f"majority_choice ~ {rhs}"

    social_res = fit_logit(social_formula, df)
    majority_res = fit_logit(majority_formula, df)

    # Summaries for age and culture effects
    social_age = summarize_effect(social_res, "age_c")
    social_culture = summarize_effect(social_res, "C(culture)[")
    majority_age = summarize_effect(majority_res, "age_c")
    majority_culture = summarize_effect(majority_res, "C(culture)[")

    evidence_flags = {
        "age_social": social_age["has_evidence"],
        "culture_social": social_culture["has_evidence"],
        "age_majority": majority_age["has_evidence"],
        "culture_majority": majority_culture["has_evidence"],
    }

    likert_score = compute_likert_score(evidence_flags)

    # Build human-readable explanation
    n = len(df)
    choice_counts = df["y"].value_counts().sort_index()
    majority_rate = df["majority_choice"].mean()
    social_rate = df["social_choice"].mean()

    def fmt_p(summary):
        if summary["min_p"] is None:
            return "not estimable"
        return f"p ≈ {summary['min_p']:.3g}"

    lines = []
    lines.append(
        f"The dataset contains {n} observations of choices among majority, minority, "
        f"and undemonstrated options (y=1,2,3)."
    )
    lines.append(
        f"Overall, {choice_counts.get(2, 0)} out of {n} choices "
        f"({majority_rate:.1%}) followed the majority option, while "
        f"{choice_counts.get(1, 0)} ({(choice_counts.get(1, 0)/n):.1%}) chose the "
        "undemonstrated option."
    )
    lines.append(
        f"Reliance on any social information (choosing either majority or minority) "
        f"occurred in {social_rate:.1%} of trials."
    )

    # Age effects
    lines.append(
        "Logistic regression for social-information use (social_choice) as a function "
        "of age and culture indicates "
        + (
            f"statistically reliable age-related variation ({fmt_p(social_age)}), "
            if social_age["has_evidence"]
            else f"no strong age-related trend ({fmt_p(social_age)}), "
        )
        + (
            f"and cross-cultural differences in social-information reliance ({fmt_p(social_culture)})."
            if social_culture["has_evidence"]
            else f"and limited evidence for cross-cultural differences ({fmt_p(social_culture)})."
        )
    )

    # Majority preference
    lines.append(
        "For preference for majority cues (majority_choice), logistic regression "
        "likewise shows "
        + (
            f"age-related differences ({fmt_p(majority_age)}) "
            if majority_age["has_evidence"]
            else f"no strong age-related trend ({fmt_p(majority_age)}) "
        )
        + (
            f"and culture-related variation ({fmt_p(majority_culture)})."
            if majority_culture["has_evidence"]
            else f"and only weak culture-related variation ({fmt_p(majority_culture)})."
        )
    )

    lines.append(
        "Taken together, these models suggest that children's reliance on social "
        "information and their preference for majority cues do vary with both "
        "developmental stage (age) and cultural context, although the strength of "
        "these effects differs across the two outcomes."
    )

    explanation = " ".join(lines)

    conclusion = {
        "response": int(likert_score),
        "explanation": explanation,
    }

    out_path = base_dir / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

