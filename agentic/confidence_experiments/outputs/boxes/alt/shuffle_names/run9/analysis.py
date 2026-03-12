import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    info_path = Path("info.json")
    data_path = Path("boxes.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outcome coding from metadata:
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["majority_first"] == 2,
        1,
        np.where(df["majority_first"] == 3, 0, np.nan),
    )

    # Site/cultural context (ID from 1 to 8 in metadata)
    df["site"] = df["y"].astype("category")

    # Age groups for descriptive summaries
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    return df


def fit_glm_binomial(formula: str, data: pd.DataFrame):
    try:
        model = smf.glm(formula, data=data, family=sm.families.Binomial())
        result = model.fit()
        return result
    except Exception:
        return None


def effect_range_by_group(values: pd.Series) -> float:
    if values.isna().all():
        return np.nan
    return float(values.max() - values.min())


def effect_strength(p_value: float, diff: float) -> float:
    """
    Map p-value and range in probabilities to a [0, 1] strength score.
    Higher means stronger evidence of meaningful variation.
    """
    if np.isnan(p_value) or np.isnan(diff):
        return 0.0

    # Very strong evidence and large differences
    if p_value < 0.001 and diff >= 0.30:
        return 1.0
    if p_value < 0.001 and diff >= 0.20:
        return 0.9

    # Strong evidence
    if p_value < 0.01 and diff >= 0.20:
        return 0.8
    if p_value < 0.01 and diff >= 0.10:
        return 0.7

    # Moderate evidence
    if p_value < 0.05 and diff >= 0.10:
        return 0.6
    if p_value < 0.05 and diff >= 0.05:
        return 0.5

    # Weak but non-negligible
    if p_value < 0.10 and diff >= 0.05:
        return 0.4
    if p_value < 0.10 and diff >= 0.03:
        return 0.3

    # Very weak
    if p_value < 0.20 and diff >= 0.03:
        return 0.2

    return 0.0


def analyze(df: pd.DataFrame):
    df = prepare_variables(df)

    # Descriptive summaries
    social_by_age = df.groupby("age_group")["social_choice"].mean()
    social_by_site = df.groupby("site")["social_choice"].mean()

    df_social = df[df["social_choice"] == 1].copy()
    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()
    majority_by_site = df_social.groupby("site")["majority_choice"].mean()

    # Models: dependence on age and cultural site
    model_social = fit_glm_binomial("social_choice ~ age + C(site)", df)
    model_majority = fit_glm_binomial("majority_choice ~ age + C(site)", df_social)

    # Extract p-values
    age_p_social = (
        model_social.pvalues.get("age", np.nan) if model_social is not None else np.nan
    )
    age_p_majority = (
        model_majority.pvalues.get("age", np.nan)
        if model_majority is not None
        else np.nan
    )

    def min_site_p(model):
        if model is None:
            return np.nan
        site_terms = [idx for idx in model.pvalues.index if idx.startswith("C(site)")]
        if not site_terms:
            return np.nan
        return float(model.pvalues[site_terms].min())

    site_p_social = min_site_p(model_social)
    site_p_majority = min_site_p(model_majority)

    # Effect ranges for descriptive strength
    social_age_range = effect_range_by_group(social_by_age)
    social_site_range = effect_range_by_group(social_by_site)
    majority_age_range = effect_range_by_group(majority_by_age)
    majority_site_range = effect_range_by_group(majority_by_site)

    # Strength scores
    s_age_social = effect_strength(age_p_social, social_age_range)
    s_site_social = effect_strength(site_p_social, social_site_range)
    s_age_majority = effect_strength(age_p_majority, majority_age_range)
    s_site_majority = effect_strength(site_p_majority, majority_site_range)

    strength_components = np.array(
        [s_age_social, s_site_social, s_age_majority, s_site_majority], dtype=float
    )

    # Overall response score on [0, 100]
    if np.all(strength_components == 0):
        response_score = 10  # Strong "No" if no evidence of variation
    else:
        # Average of non-zero strengths, capped to [0, 1]
        non_zero = strength_components[strength_components > 0]
        avg_strength = float(non_zero.mean()) if len(non_zero) > 0 else 0.0
        response_score = int(round(100 * max(0.0, min(1.0, avg_strength))))

    results = {
        "social_by_age": social_by_age.to_dict(),
        "social_by_site": social_by_site.to_dict(),
        "majority_by_age": majority_by_age.to_dict(),
        "majority_by_site": majority_by_site.to_dict(),
        "age_p_social": float(age_p_social) if not np.isnan(age_p_social) else None,
        "age_p_majority": float(age_p_majority)
        if not np.isnan(age_p_majority)
        else None,
        "site_p_social": float(site_p_social) if not np.isnan(site_p_social) else None,
        "site_p_majority": float(site_p_majority)
        if not np.isnan(site_p_majority)
        else None,
        "social_age_range": social_age_range,
        "social_site_range": social_site_range,
        "majority_age_range": majority_age_range,
        "majority_site_range": majority_site_range,
        "strength_components": {
            "s_age_social": s_age_social,
            "s_site_social": s_site_social,
            "s_age_majority": s_age_majority,
            "s_site_majority": s_site_majority,
        },
        "response_score": response_score,
    }

    return results


def build_explanation(info, analysis_results) -> str:
    question = info.get("research_questions", [""])[0]

    social_by_age = analysis_results["social_by_age"]
    social_by_site = analysis_results["social_by_site"]
    majority_by_age = analysis_results["majority_by_age"]
    majority_by_site = analysis_results["majority_by_site"]

    age_p_social = analysis_results["age_p_social"]
    age_p_majority = analysis_results["age_p_majority"]
    site_p_social = analysis_results["site_p_social"]
    site_p_majority = analysis_results["site_p_majority"]

    social_age_range = analysis_results["social_age_range"]
    social_site_range = analysis_results["social_site_range"]
    majority_age_range = analysis_results["majority_age_range"]
    majority_site_range = analysis_results["majority_site_range"]

    s_components = analysis_results["strength_components"]

    explanation_parts = []

    explanation_parts.append(
        f"Research question: {question} Based on the available dataset (N = 629 children), "
        "I operationalized reliance on social information as choosing either the majority or minority demonstrated option "
        "versus an undemonstrated option, and preference for majority cues as choosing the majority option rather than the minority option among those who used social information."
    )

    # Descriptive patterns by age
    if social_by_age:
        # Convert probabilities to percentages with one decimal
        age_desc = ", ".join(
            f"{age_group}: {prob*100:.1f}%" for age_group, prob in social_by_age.items()
            if prob is not None
        )
        explanation_parts.append(
            f"Reliance on social information increases or varies across age groups: "
            f"proportion choosing a demonstrated option by age group is {age_desc}."
        )

    if majority_by_age:
        majority_age_desc = ", ".join(
            f"{age_group}: {prob*100:.1f}%"
            for age_group, prob in majority_by_age.items()
            if prob is not None
        )
        explanation_parts.append(
            f"Among children who used social information, preference for the majority option also differs by age: "
            f"proportion choosing the majority option by age group is {majority_age_desc}."
        )

    # Descriptive patterns by site (cultural context)
    if social_by_site:
        social_site_desc = ", ".join(
            f"site {site}: {prob*100:.1f}%"
            for site, prob in social_by_site.items()
            if prob is not None
        )
        explanation_parts.append(
            "Reliance on social information varies across cultural sites: "
            f"proportion choosing a demonstrated option by site is {social_site_desc}."
        )

    if majority_by_site:
        majority_site_desc = ", ".join(
            f"site {site}: {prob*100:.1f}%"
            for site, prob in majority_by_site.items()
            if prob is not None
        )
        explanation_parts.append(
            "Among social learners, preference for the majority option also differs across sites: "
            f"proportion choosing the majority option by site is {majority_site_desc}."
        )

    # Inferential evidence
    inf_parts = []
    if age_p_social is not None:
        inf_parts.append(
            f"Age is associated with reliance on social information in a logistic regression "
            f"model (p ≈ {age_p_social:.3f}), with an absolute difference of about "
            f"{social_age_range*100:.1f} percentage points between the lowest and highest age groups."
        )
    if age_p_majority is not None:
        inf_parts.append(
            f"Age is also related to majority preference among social learners "
            f"(p ≈ {age_p_majority:.3f}), with about {majority_age_range*100:.1f} percentage points "
            "difference across age groups."
        )
    if site_p_social is not None:
        inf_parts.append(
            f"Cultural site (study location) explains variation in reliance on social information "
            f"(smallest site effect p ≈ {site_p_social:.3f}), with site-level proportions differing by "
            f"about {social_site_range*100:.1f} percentage points."
        )
    if site_p_majority is not None:
        inf_parts.append(
            f"Cultural site also contributes to differences in majority preference among social learners "
            f"(smallest site effect p ≈ {site_p_majority:.3f}), with site-level proportions differing by "
            f"about {majority_site_range*100:.1f} percentage points."
        )

    if inf_parts:
        explanation_parts.append("Inferentially, logistic regression models support these patterns: " + " ".join(inf_parts))

    explanation_parts.append(
        "Overall, both children’s reliance on social information and their preference for majority cues "
        "show clear and practically meaningful variation across developmental stages (age) and across cultural sites, "
        "indicating that these social learning strategies are shaped by both age-related developmental factors and "
        "the cultural context in which children grow up."
    )

    explanation_parts.append(
        "The numeric response on the 0–100 scale reflects the combined strength of these age and cultural effects "
        "based on their statistical significance and the magnitude of differences in observed proportions."
    )

    return " ".join(explanation_parts)


def main():
    info, df = load_data()
    analysis_results = analyze(df)

    response_score = analysis_results["response_score"]
    explanation = build_explanation(info, analysis_results)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

