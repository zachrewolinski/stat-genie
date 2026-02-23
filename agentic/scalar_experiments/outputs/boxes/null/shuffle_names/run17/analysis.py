import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def load_metadata():
    info_path = Path("info.json")
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    df = pd.read_csv("boxes.csv")

    # According to the metadata:
    # - "majority_first" column actually encodes the outcome:
    #   1 = undemonstrated (no social reliance), 2 = majority, 3 = minority.
    # - "culture" column description refers to whether the majority option
    #   was demonstrated first (order manipulation).
    # - "y" is the site ID (cultural context).
    df = df.copy()
    df["outcome"] = df["majority_first"]

    # Social reliance: 1 if child chose any demonstrated option (majority/minority), 0 if undemonstrated.
    df["social_reliance"] = np.where(df["outcome"] == 1, 0, 1)

    # Majority preference among children who used social information.
    social_mask = df["social_reliance"] == 1
    df["majority_choice"] = np.where(
        social_mask & (df["outcome"] == 2),
        1,
        np.where(social_mask & (df["outcome"] == 3), 0, np.nan),
    )

    # Rename columns for clarity in models.
    df["age_years"] = df["age"]
    df["site"] = df["y"].astype("category")
    df["majority_first_order"] = df["culture"]

    return df


def fit_logistic(formula, data, family=sm.families.Binomial()):
    model = smf.glm(formula=formula, data=data, family=family)
    result = model.fit()
    return result


def likelihood_ratio_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    if df_diff <= 0:
        lr_pvalue = np.nan
    else:
        lr_pvalue = 1 - chi2.cdf(lr_stat, df_diff)
    return lr_stat, lr_pvalue


def analyse_social_reliance(df):
    # Full model: social reliance predicted by age and site, controlling for demonstration order.
    full = fit_logistic("social_reliance ~ age_years + C(site) + majority_first_order", df)
    # Reduced models to test specific factors.
    reduced_no_site = fit_logistic("social_reliance ~ age_years + majority_first_order", df)
    reduced_no_age = fit_logistic("social_reliance ~ C(site) + majority_first_order", df)

    lr_site_stat, lr_site_p = likelihood_ratio_test(full, reduced_no_site)
    lr_age_stat, lr_age_p = likelihood_ratio_test(full, reduced_no_age)

    age_coef = full.params.get("age_years", np.nan)

    return {
        "model_summary": full.summary().as_text(),
        "lr_site_stat": float(lr_site_stat),
        "lr_site_p": float(lr_site_p),
        "lr_age_stat": float(lr_age_stat),
        "lr_age_p": float(lr_age_p),
        "age_coef": float(age_coef) if np.isfinite(age_coef) else None,
    }


def analyse_majority_preference(df):
    majority_df = df[df["majority_choice"].notna()].copy()
    if majority_df.empty:
        return {
            "model_summary": "No data with social reliance for majority/minority choices.",
            "lr_site_stat": np.nan,
            "lr_site_p": np.nan,
            "lr_age_stat": np.nan,
            "lr_age_p": np.nan,
            "age_coef": None,
        }

    full = fit_logistic("majority_choice ~ age_years + C(site) + majority_first_order", majority_df)
    reduced_no_site = fit_logistic("majority_choice ~ age_years + majority_first_order", majority_df)
    reduced_no_age = fit_logistic("majority_choice ~ C(site) + majority_first_order", majority_df)

    lr_site_stat, lr_site_p = likelihood_ratio_test(full, reduced_no_site)
    lr_age_stat, lr_age_p = likelihood_ratio_test(full, reduced_no_age)

    age_coef = full.params.get("age_years", np.nan)

    return {
        "model_summary": full.summary().as_text(),
        "lr_site_stat": float(lr_site_stat),
        "lr_site_p": float(lr_site_p),
        "lr_age_stat": float(lr_age_stat),
        "lr_age_p": float(lr_age_p),
        "age_coef": float(age_coef) if np.isfinite(age_coef) else None,
    }


def derive_conclusion(social_results, majority_results):
    social_age_p = social_results["lr_age_p"]
    social_site_p = social_results["lr_site_p"]
    social_age_coef = social_results["age_coef"]

    majority_age_p = majority_results["lr_age_p"]
    majority_site_p = majority_results["lr_site_p"]
    majority_age_coef = majority_results["age_coef"]

    # Determine evidence strength using simple thresholds.
    def evidence_level(p):
        if p is None or np.isnan(p):
            return "none"
        if p < 0.001:
            return "very_strong"
        if p < 0.01:
            return "strong"
        if p < 0.05:
            return "moderate"
        if p < 0.1:
            return "weak"
        return "none"

    social_age_evidence = evidence_level(social_age_p)
    social_site_evidence = evidence_level(social_site_p)
    majority_age_evidence = evidence_level(majority_age_p)
    majority_site_evidence = evidence_level(majority_site_p)

    any_developmental_effect = social_age_evidence != "none" or majority_age_evidence != "none"
    any_cultural_effect = social_site_evidence != "none" or majority_site_evidence != "none"

    # Map combined evidence to Likert-style 0–100 score.
    if any_developmental_effect or any_cultural_effect:
        # Start from a baseline "yes" strength and adjust by evidence.
        score = 0
        for ev in [
            social_age_evidence,
            social_site_evidence,
            majority_age_evidence,
            majority_site_evidence,
        ]:
            if ev == "very_strong":
                score += 30
            elif ev == "strong":
                score += 22
            elif ev == "moderate":
                score += 15
            elif ev == "weak":
                score += 8
        score = min(score, 100)
        if score < 50:
            score = 50
        response_scalar = int(round(score))
    else:
        # No reliable evidence of differences.
        response_scalar = 20

    # Build human-readable explanation summarising key statistical evidence.
    explanation_lines = []
    explanation_lines.append(
        "I modelled children’s social reliance (choosing any demonstrated option vs. an undemonstrated one) "
        "and majority preference (choosing the majority demonstrator vs. the minority demonstrator) "
        "using logistic regression with age and cultural site as predictors, controlling for whether the majority "
        "option was demonstrated first."
    )

    explanation_lines.append(
        f"For social reliance, likelihood-ratio tests indicated an age effect p={social_age_p:.4f} "
        f"and a cultural site effect p={social_site_p:.4f}."
    )
    explanation_lines.append(
        f"For majority preference among children who used social information, age effect p={majority_age_p:.4f} "
        f"and cultural site effect p={majority_site_p:.4f}."
    )

    if social_age_coef is not None and np.isfinite(social_age_coef):
        direction = "increased" if social_age_coef > 0 else "decreased"
        explanation_lines.append(
            f"The age coefficient for social reliance ({social_age_coef:.3f}) suggests that with age children’s "
            f"tendency to rely on social information generally {direction}."
        )

    if majority_age_coef is not None and np.isfinite(majority_age_coef):
        direction = "increased" if majority_age_coef > 0 else "decreased"
        explanation_lines.append(
            f"The age coefficient for majority preference ({majority_age_coef:.3f}) indicates that, among children "
            f"who copy demonstrators, preference for the majority option {direction} with age."
        )

    if response_scalar >= 50:
        explanation_lines.append(
            "Taken together, these results provide statistical evidence that both reliance on social information "
            "and preference for majority cues vary across cultures and developmental stages in this dataset."
        )
    else:
        explanation_lines.append(
            "Taken together, the models do not provide strong evidence that reliance on social information or "
            "preference for majority cues differ systematically across cultures or developmental stages in this dataset."
        )

    explanation = " ".join(explanation_lines)
    return response_scalar, explanation


def main():
    df = load_data()
    _ = load_metadata()  # Not strictly needed for modelling, but kept for completeness.

    social_results = analyse_social_reliance(df)
    majority_results = analyse_majority_preference(df)

    response_scalar, explanation = derive_conclusion(social_results, majority_results)

    conclusion = {"response": int(response_scalar), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
