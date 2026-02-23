import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def analyze_social_reliance(df: pd.DataFrame):
    """
    Model whether a child relies on social information at all
    (choosing majority or minority vs undemonstrated option).
    """
    df = df.copy()
    df["social_reliance"] = np.where(df["y"].isin([2, 3]), 1, 0)

    # Logistic regression with age (continuous) and culture (categorical)
    model = smf.logit("social_reliance ~ age + C(culture)", data=df).fit(disp=False)

    # Likelihood ratio tests for age and culture terms
    # Null model without age
    model_no_age = smf.logit("social_reliance ~ C(culture)", data=df).fit(disp=False)
    lr_stat_age = 2 * (model.llf - model_no_age.llf)
    df_age = model.df_model - model_no_age.df_model
    p_age = chi2.sf(lr_stat_age, df_age) if df_age > 0 else np.nan

    # Null model without culture
    model_no_culture = smf.logit("social_reliance ~ age", data=df).fit(disp=False)
    lr_stat_culture = 2 * (model.llf - model_no_culture.llf)
    df_culture = model.df_model - model_no_culture.df_model
    p_culture = chi2.sf(lr_stat_culture, df_culture) if df_culture > 0 else np.nan

    return {
        "model": model,
        "p_age": float(p_age),
        "p_culture": float(p_culture),
    }


def analyze_majority_preference(df: pd.DataFrame):
    """
    Among children who relied on social information (chose majority or minority),
    test whether preference for the majority vs minority varies with age and culture.
    """
    df = df[df["y"].isin([2, 3])].copy()
    # 1 = majority, 0 = minority
    df["majority_choice"] = np.where(df["y"] == 2, 1, 0)

    model = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)

    model_no_age = smf.logit("majority_choice ~ C(culture)", data=df).fit(disp=False)
    lr_stat_age = 2 * (model.llf - model_no_age.llf)
    df_age = model.df_model - model_no_age.df_model
    p_age = chi2.sf(lr_stat_age, df_age) if df_age > 0 else np.nan

    model_no_culture = smf.logit("majority_choice ~ age", data=df).fit(disp=False)
    lr_stat_culture = 2 * (model.llf - model_no_culture.llf)
    df_culture = model.df_model - model_no_culture.df_model
    p_culture = chi2.sf(lr_stat_culture, df_culture) if df_culture > 0 else np.nan

    return {
        "model": model,
        "p_age": float(p_age),
        "p_culture": float(p_culture),
    }


def interpret_results(p_vals, alpha=0.05):
    """Summarize evidence for variation across age and culture."""
    age_sig = p_vals["age"] < alpha
    culture_sig = p_vals["culture"] < alpha

    if age_sig and culture_sig:
        desc = "strong_evidence_both"
    elif age_sig or culture_sig:
        desc = "partial_evidence"
    else:
        desc = "no_evidence"

    return desc, age_sig, culture_sig


def map_to_likert(evidence_desc: str) -> int:
    """
    Map overall evidence about variation (any variation across cultures or age)
    to a 0–100 Likert scale where higher = stronger 'Yes, there is variation'.
    """
    if evidence_desc == "strong_evidence_both":
        return 85
    if evidence_desc == "partial_evidence":
        return 65
    if evidence_desc == "no_evidence":
        return 20
    return 50


def main():
    df = load_data(Path("boxes.csv"))

    social_res = analyze_social_reliance(df)
    majority_res = analyze_majority_preference(df)

    # Collect key statistics
    pvals = {
        "social_age": social_res["p_age"],
        "social_culture": social_res["p_culture"],
        "majority_age": majority_res["p_age"],
        "majority_culture": majority_res["p_culture"],
    }

    # Combine evidence: if any of the four tests show significance,
    # we treat this as evidence that reliance/preference varies.
    combined = {
        "age": min(pvals["social_age"], pvals["majority_age"]),
        "culture": min(pvals["social_culture"], pvals["majority_culture"]),
    }

    evidence_desc, age_sig, culture_sig = interpret_results(
        {"age": combined["age"], "culture": combined["culture"]}
    )
    likert_score = map_to_likert(evidence_desc)

    # Build human-readable explanation
    explanation_lines = []
    explanation_lines.append(
        "I tested whether children’s reliance on social information "
        "(choosing demonstrated options vs an undemonstrated one) and their "
        "preference for majority over minority demonstrators varied with age "
        "and across eight cultural sites."
    )
    explanation_lines.append(
        f"For social reliance (any demonstrated choice), likelihood-ratio tests "
        f"for age (p = {social_res['p_age']:.4f}) and culture "
        f"(p = {social_res['p_culture']:.4f}) were obtained from logistic regression "
        f"models including age and culture predictors."
    )
    explanation_lines.append(
        f"For majority vs minority choice among children who used social information, "
        f"age (p = {majority_res['p_age']:.4f}) and culture "
        f"(p = {majority_res['p_culture']:.4f}) were similarly tested."
    )

    if evidence_desc == "strong_evidence_both":
        explanation_lines.append(
            "Both age and culture showed statistically significant associations "
            "with at least one of these outcomes (p < 0.05), indicating clear "
            "developmental and cross-cultural differences in how children rely on "
            "social information and follow majority cues."
        )
    elif evidence_desc == "partial_evidence":
        explanation_lines.append(
            "At least one of age or culture showed statistically significant "
            "associations with these outcomes (p < 0.05), providing evidence that "
            "reliance on social information or preference for majority cues varies "
            "across developmental stage or cultural context."
        )
    else:
        explanation_lines.append(
            "Neither age nor culture showed statistically significant associations "
            "with these outcomes (p ≥ 0.05), suggesting limited evidence for "
            "developmental or cross-cultural variation in reliance on social "
            "information or majority preference in this dataset."
        )

    variation_phrase = (
        "do not vary across cultures and developmental stages"
        if evidence_desc == "no_evidence"
        else "do vary across cultures and developmental stages"
    )
    explanation_lines.append(
        f"On the 0–100 scale (0 = strong 'No', 100 = strong 'Yes'), "
        f"I map this pattern of evidence to a response of {likert_score}, "
        f"reflecting the overall strength of evidence that children’s reliance "
        f"on social information and preference for majority cues {variation_phrase}."
    )

    conclusion = {
        "response": int(likert_score),
        "explanation": " ".join(explanation_lines),
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
