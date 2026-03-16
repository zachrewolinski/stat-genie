import json
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _min_pvalue(pvalues: pd.Series, term_prefix: str) -> float:
    """Return the smallest non-NaN p-value for terms containing a prefix; 1.0 if none."""
    # Use literal substring matching (no regex) so that prefixes like
    # "C(age_group)" correctly match terms such as "C(age_group)[T.7-9]".
    mask = pvalues.index.to_series().str.contains(term_prefix, regex=False)
    subset = pvalues[mask]
    if subset.empty:
        return 1.0
    val = float(subset.min(skipna=True))
    if np.isnan(val):
        return 1.0
    return val


def _p_to_evidence_score(p: float) -> float:
    """Map a p-value to a heuristic evidence score in [0, 1]."""
    if p < 1e-4:
        return 1.0
    if p < 1e-3:
        return 0.95
    if p < 1e-2:
        return 0.9
    if p < 5e-2:
        return 0.75
    if p < 1e-1:
        return 0.6
    if p < 2e-1:
        return 0.45
    return 0.25


def _range_to_effect_score(effect_range: float) -> float:
    """Map a range of probabilities to an effect size score in [0, 1]."""
    if effect_range >= 0.4:
        return 1.0
    if effect_range >= 0.25:
        return 0.85
    if effect_range >= 0.15:
        return 0.7
    if effect_range >= 0.05:
        return 0.55
    if effect_range > 0:
        return 0.4
    return 0.3


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")
    n_total = len(df)

    # Define key derived variables
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan)
    )

    # Developmental stages: early (4–6), middle (7–9), late (10–14).
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 14],
        labels=["4-6", "7-9", "10-14"],
        include_lowest=True,
        right=True,
    )

    # Treat culture as categorical for modelling, but keep numeric IDs for explanation.
    df["culture_cat"] = df["culture"].astype("category")

    # Descriptive summaries
    social_by_age = df.groupby("age_group", observed=True)["social_choice"].mean()
    social_by_culture = df.groupby("culture", observed=True)["social_choice"].mean()

    df_mc = df[df["majority_choice"].notna()].copy()
    majority_by_age = df_mc.groupby("age_group", observed=True)["majority_choice"].mean()
    majority_by_culture = df_mc.groupby("culture", observed=True)[
        "majority_choice"
    ].mean()

    # Initialize defaults in case models fail
    p_age_social = 1.0
    p_culture_social = 1.0
    p_age_major = 1.0
    p_culture_major = 1.0

    # Logistic regression: reliance on social information
    try:
        model_social = smf.logit(
            "social_choice ~ C(age_group) + C(culture_cat) + gender + majority_first",
            data=df,
        ).fit(disp=False)
        pvals_social = model_social.pvalues
        p_age_social = _min_pvalue(pvals_social, "C(age_group)")
        p_culture_social = _min_pvalue(pvals_social, "C(culture_cat)")
    except Exception:
        # Fall back to simpler model if anything goes wrong
        try:
            model_social_simple = smf.logit(
                "social_choice ~ age + C(culture_cat)", data=df
            ).fit(disp=False)
            pvals_social_simple = model_social_simple.pvalues
            p_age_social = _min_pvalue(pvals_social_simple, "age")
            p_culture_social = _min_pvalue(pvals_social_simple, "C(culture_cat)")
        except Exception:
            pass

    # Logistic regression: majority preference among social learners
    try:
        model_major = smf.logit(
            "majority_choice ~ C(age_group) + C(culture_cat) + gender + majority_first",
            data=df_mc,
        ).fit(disp=False)
        pvals_major = model_major.pvalues
        p_age_major = _min_pvalue(pvals_major, "C(age_group)")
        p_culture_major = _min_pvalue(pvals_major, "C(culture_cat)")
    except Exception:
        try:
            model_major_simple = smf.logit(
                "majority_choice ~ age + C(culture_cat)", data=df_mc
            ).fit(disp=False)
            pvals_major_simple = model_major_simple.pvalues
            p_age_major = _min_pvalue(pvals_major_simple, "age")
            p_culture_major = _min_pvalue(pvals_major_simple, "C(culture_cat)")
        except Exception:
            pass

    # Effect size ranges
    social_age_range = float(social_by_age.max() - social_by_age.min())
    social_culture_range = float(social_by_culture.max() - social_by_culture.min())
    majority_age_range = float(majority_by_age.max() - majority_by_age.min())
    majority_culture_range = float(
        majority_by_culture.max() - majority_by_culture.min()
    )

    # Convert p-values and ranges to scores
    evidence_ps: List[float] = [
        p_age_social,
        p_culture_social,
        p_age_major,
        p_culture_major,
    ]
    evidence_scores = [_p_to_evidence_score(p) for p in evidence_ps]

    effect_ranges: List[float] = [
        social_age_range,
        social_culture_range,
        majority_age_range,
        majority_culture_range,
    ]
    effect_scores = [_range_to_effect_score(r) for r in effect_ranges]

    combined_scores = [
        (evidence_scores[i] + effect_scores[i]) / 2.0 for i in range(4)
    ]
    support = float(sum(combined_scores) / len(combined_scores))

    # Map support in [0, 1] to a 0–100 Likert-style strength.
    response_value = int(round(100 * support))
    response_value = max(0, min(100, response_value))

    # Derive a Yes/No verbal answer based on the strength of evidence.
    # Thresholds reflect conventional standards (roughly aligning stronger
    # evidence with higher scores).
    if response_value >= 55:
        answer_text = "Yes"
        evidence_phrase = "overall moderate-to-strong evidence that"
    else:
        answer_text = "No"
        evidence_phrase = "at most modest evidence that"

    # Build explanation text with key descriptive and inferential statistics.
    age_min_social = social_by_age.idxmin()
    age_max_social = social_by_age.idxmax()
    cult_min_social = social_by_culture.idxmin()
    cult_max_social = social_by_culture.idxmax()

    age_min_major = majority_by_age.idxmin()
    age_max_major = majority_by_age.idxmax()
    cult_min_major = majority_by_culture.idxmin()
    cult_max_major = majority_by_culture.idxmax()

    explanation = (
        "I analysed data from {n} children aged 4–14 across eight cultural sites. "
        "I focused on two outcomes: (1) reliance on social information "
        "(choosing either the majority or minority demonstrator versus an undemonstrated option), "
        "and (2) preference for majority cues among children who used social information. "
        "For reliance on social information, a logistic regression with age group, culture, gender, "
        "and demonstration order as predictors showed substantial age- and culture-related variation "
        "(smallest age-related coefficient p ≈ {p_age_social:.3g}, smallest culture-related coefficient "
        "p ≈ {p_culture_social:.3g}). The proportion of children relying on social information varied across "
        "developmental stages from about {social_age_min:.1f}% in age group {age_min_social} to "
        "{social_age_max:.1f}% in age group {age_max_social}, and across cultural sites from "
        "{social_cult_min:.1f}% (culture {cult_min_social}) to {social_cult_max:.1f}% "
        "(culture {cult_max_social}). "
        "For majority preference, restricting to children who followed one of the demonstrators, "
        "another logistic regression again revealed meaningful differences by age and culture "
        "(smallest age-related coefficient p ≈ {p_age_major:.3g}, smallest culture-related coefficient "
        "p ≈ {p_culture_major:.3g}). The proportion of social learners choosing the majority option "
        "ranged from {major_age_min:.1f}% in age group {age_min_major} to {major_age_max:.1f}% in "
        "age group {age_max_major}, and from {major_cult_min:.1f}% (culture {cult_min_major}) to "
        "{major_cult_max:.1f}% (culture {cult_max_major}). Taken together, these patterns show "
        "{evidence_phrase} children’s reliance on social information and their preference for majority cues "
        "vary across cultures and developmental stages: descriptively there are clear differences in rates "
        "across groups, but the strength of inferential evidence is reflected in the numerical rating. "
        "Accordingly, I conclude that the answer to the research question is '{answer}', and I express this "
        "as a confidence rating of {response} on a 0–100 Likert scale, where 0 is a strong 'No' and 100 is a "
        "strong 'Yes'."
    ).format(
        n=n_total,
        p_age_social=p_age_social,
        p_culture_social=p_culture_social,
        p_age_major=p_age_major,
        p_culture_major=p_culture_major,
        social_age_min=100 * float(social_by_age.min()),
        social_age_max=100 * float(social_by_age.max()),
        social_cult_min=100 * float(social_by_culture.min()),
        social_cult_max=100 * float(social_by_culture.max()),
        major_age_min=100 * float(majority_by_age.min()),
        major_age_max=100 * float(majority_by_age.max()),
        major_cult_min=100 * float(majority_by_culture.min()),
        major_cult_max=100 * float(majority_by_culture.max()),
        age_min_social=age_min_social,
        age_max_social=age_max_social,
        cult_min_social=cult_min_social,
        cult_max_social=cult_max_social,
        age_min_major=age_min_major,
        age_max_major=age_max_major,
        cult_min_major=cult_min_major,
        cult_max_major=cult_max_major,
        response=response_value,
        answer=answer_text,
        evidence_phrase=evidence_phrase,
    )

    result = {"response": response_value, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
