import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_metadata():
    info_path = Path("info.json")
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    return pd.read_csv("boxes.csv")


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Reliance on any social information: chose a demonstrated option (majority/minority) vs undemonstrated
    df["social_info"] = (df["y"] != 1).astype(int)

    # Majority preference among children who followed any demonstrator
    mask_demo = df["y"].isin([2, 3])
    df_majority = df.loc[mask_demo].copy()
    df_majority["majority_choice"] = (df_majority["y"] == 2).astype(int)

    return df, df_majority


def fit_logistic_models(df_all: pd.DataFrame, df_majority: pd.DataFrame):
    results = {}

    # Model 1: reliance on social information (any demonstrator vs undemonstrated option)
    try:
        model_social = smf.logit(
            formula="social_info ~ age + C(culture)",
            data=df_all,
        ).fit(disp=False)
        results["social"] = model_social
    except Exception as exc:  # pragma: no cover - robust fallback
        results["social_error"] = str(exc)

    # Model 2: preference for majority vs minority among those using social info
    try:
        model_majority = smf.logit(
            formula="majority_choice ~ age + C(culture)",
            data=df_majority,
        ).fit(disp=False)
        results["majority"] = model_majority
    except Exception as exc:  # pragma: no cover
        results["majority_error"] = str(exc)

    return results


def summarize_effects(model, age_var: str = "age"):
    pvalues = model.pvalues

    age_p = float(pvalues.get(age_var, np.nan))

    culture_terms = [name for name in pvalues.index if name.startswith("C(culture)")]
    culture_ps = [float(pvalues[name]) for name in culture_terms]
    culture_min_p = float(np.min(culture_ps)) if culture_ps else np.nan

    return {
        "age_p": age_p,
        "culture_min_p": culture_min_p,
    }


def compute_predicted_differences(model, df: pd.DataFrame, outcome_col: str):
    # Age effect: predicted probabilities at lower vs higher age quantiles
    age_quantiles = df["age"].quantile([0.25, 0.75]).to_dict()
    cultures = sorted(df["culture"].unique())
    baseline_culture = cultures[0]

    scenarios = []
    for label, age_val in age_quantiles.items():
        scenarios.append(
            {
                "age": age_val,
                "culture": baseline_culture,
            }
        )

    scenario_df = pd.DataFrame(scenarios)
    preds = model.predict(scenario_df)
    age_diff = float(preds.max() - preds.min())

    # Culture effect: predicted probabilities across cultures at median age
    median_age = float(df["age"].median())
    culture_scenarios = pd.DataFrame(
        {
            "age": [median_age] * len(cultures),
            "culture": cultures,
        }
    )
    culture_preds = model.predict(culture_scenarios)
    culture_diff = float(culture_preds.max() - culture_preds.min())

    return {
        "age_diff": age_diff,
        "culture_diff": culture_diff,
        "pred_quantiles": {
            "age_low": float(age_quantiles[0.25]),
            "age_high": float(age_quantiles[0.75]),
        },
    }


def determine_likert_score(social_summary, majority_summary):
    # Start from neutral evidence
    score = 50

    strong_evidence = 0

    for summary in (social_summary, majority_summary):
        if summary is None:
            continue

        age_p = summary.get("age_p")
        culture_min_p = summary.get("culture_min_p")
        age_diff = summary.get("age_diff", 0.0)
        culture_diff = summary.get("culture_diff", 0.0)

        # Evidence thresholds for significance and effect size
        if age_p is not None and not np.isnan(age_p) and age_p < 0.05 and abs(age_diff) > 0.05:
            strong_evidence += 1
        if culture_min_p is not None and not np.isnan(culture_min_p) and culture_min_p < 0.05 and abs(culture_diff) > 0.05:
            strong_evidence += 1

    if strong_evidence == 0:
        score = 30
    elif strong_evidence == 1:
        score = 60
    elif strong_evidence == 2:
        score = 75
    elif strong_evidence >= 3:
        score = 90

    score = int(max(0, min(100, score)))
    return score


def main():
    info = load_metadata()
    df_raw = load_data()

    df_all, df_majority = prepare_variables(df_raw)
    models = fit_logistic_models(df_all, df_majority)

    social_summary = None
    majority_summary = None

    explanation_parts = []
    research_question = info.get("research_questions", [""])[0]

    if "social" in models:
        social_model = models["social"]
        social_stats = summarize_effects(social_model)
        social_effects = compute_predicted_differences(social_model, df_all, "social_info")
        social_summary = {**social_stats, **social_effects}
        explanation_parts.append(
            "For reliance on any social information (choosing a demonstrated option vs an undemonstrated one), "
            f"the logistic model with age and culture predictors showed an age p-value of {social_stats['age_p']:.3f} "
            f"and a minimum culture-related p-value of {social_stats['culture_min_p']:.3f}. "
            f"Predicted probabilities differed by about {social_effects['age_diff']:.2f} across lower vs higher ages "
            f"and by about {social_effects['culture_diff']:.2f} across cultures at the median age."
        )

    if "majority" in models:
        majority_model = models["majority"]
        majority_stats = summarize_effects(majority_model)
        majority_effects = compute_predicted_differences(majority_model, df_majority, "majority_choice")
        majority_summary = {**majority_stats, **majority_effects}
        explanation_parts.append(
            "For preference between majority vs minority demonstrators (among children who used any social information), "
            f"the logistic model with age and culture predictors yielded an age p-value of {majority_stats['age_p']:.3f} "
            f"and a minimum culture-related p-value of {majority_stats['culture_min_p']:.3f}. "
            f"Predicted majority-choice probabilities changed by about {majority_effects['age_diff']:.2f} across age "
            f"and by about {majority_effects['culture_diff']:.2f} across cultures at the median age."
        )

    likert_score = determine_likert_score(social_summary, majority_summary)

    if likert_score >= 50:
        conclusion = "Overall, the evidence supports the claim that children's reliance on social information and their preference for majority cues vary across cultures and developmental stages."
    else:
        conclusion = "Overall, the evidence does not strongly support systematic variation in children's reliance on social information and majority preferences across cultures and developmental stages."

    explanation_intro = (
        f"Research question: '{research_question}'. "
        "I modeled the probability of using social information and the probability of following majority vs minority demonstrators "
        "as logistic functions of age and cultural site."
    )

    explanation = " ".join([explanation_intro] + explanation_parts + [conclusion])

    output = {
        "response": likert_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

