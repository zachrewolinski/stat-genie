import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


def likelihood_ratio_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(lr_stat), int(df_diff), float(p_value)


def run_analysis():
    df = pd.read_csv("boxes.csv")

    # Define key derived outcomes
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df_social = df[df["social"] == 1].copy()

    results = {}

    # Model 1: Reliance on social information (any demonstrated option vs undemonstrated)
    model_social_base = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False)
    model_social_culture = smf.logit(
        "social ~ age + gender + majority_first + C(culture)", data=df
    ).fit(disp=False)

    lr_culture_social, df_culture_social, p_culture_social = likelihood_ratio_test(
        model_social_culture, model_social_base
    )

    coef_age_social = model_social_culture.params.get("age", np.nan)
    p_age_social = model_social_culture.pvalues.get("age", np.nan)

    results["social"] = {
        "n": int(df.shape[0]),
        "prop_social": float(df["social"].mean()),
        "lr_culture": {
            "lr_stat": lr_culture_social,
            "df": df_culture_social,
            "p_value": p_culture_social,
        },
        "age_effect": {
            "coef": float(coef_age_social),
            "odds_ratio": float(np.exp(coef_age_social)),
            "p_value": float(p_age_social),
        },
    }

    # Model 2: Preference for majority vs minority among those who used social information
    model_majority_base = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=df_social
    ).fit(disp=False)
    model_majority_culture = smf.logit(
        "majority_choice ~ age + gender + majority_first + C(culture)",
        data=df_social,
    ).fit(disp=False)

    lr_culture_majority, df_culture_majority, p_culture_majority = (
        likelihood_ratio_test(model_majority_culture, model_majority_base)
    )

    coef_age_majority = model_majority_culture.params.get("age", np.nan)
    p_age_majority = model_majority_culture.pvalues.get("age", np.nan)

    results["majority"] = {
        "n": int(df_social.shape[0]),
        "prop_majority_among_social": float(df_social["majority_choice"].mean()),
        "lr_culture": {
            "lr_stat": lr_culture_majority,
            "df": df_culture_majority,
            "p_value": p_culture_majority,
        },
        "age_effect": {
            "coef": float(coef_age_majority),
            "odds_ratio": float(np.exp(coef_age_majority)),
            "p_value": float(p_age_majority),
        },
    }

    # Derive an overall conclusion based on statistical evidence
    # Criteria: evidence for variation if either culture or age effects are significant (p < 0.05)
    social_varies = (p_culture_social < 0.05) or (p_age_social < 0.05)
    majority_varies = (p_culture_majority < 0.05) or (p_age_majority < 0.05)

    if social_varies or majority_varies:
        # Some evidence that reliance on social information and/or majority preference
        # varies across cultures or with age.
        response = 75
        answer = "Yes"
    else:
        response = 25
        answer = "No"

    explanation_parts = []

    explanation_parts.append(
        "Research question: Do children's reliance on social information "
        "and preference for majority cues vary across cultures and developmental stages?"
    )
    explanation_parts.append(
        "I modelled two binary outcomes using logistic regression: "
        "(1) reliance on social information (choosing any demonstrated option vs an undemonstrated one), "
        "and (2) preference for the majority option among children who followed any demonstrator."
    )
    explanation_parts.append(
        "For reliance on social information, I regressed the binary outcome on age, gender, "
        "whether the majority was demonstrated first, and culture (as a categorical predictor) "
        "in a logistic regression. A likelihood-ratio test compared a model with culture to a "
        "model without culture, and Wald tests assessed the age effect."
    )
    explanation_parts.append(
        f"In this model, the overall effect of culture on using social information had "
        f"LR statistic {lr_culture_social:.2f} (df = {df_culture_social}), "
        f"p = {p_culture_social:.3f}, and the age coefficient was "
        f"{coef_age_social:.3f} (odds ratio {np.exp(coef_age_social):.2f}, "
        f"p = {p_age_social:.3f})."
    )
    explanation_parts.append(
        "For preference for majority cues, I restricted the sample to children who chose "
        "a demonstrated option and regressed choosing the majority vs minority option "
        "on age, gender, majority-first, and culture. I again used a likelihood-ratio "
        "test for culture and a Wald test for age."
    )
    explanation_parts.append(
        f"In this model, the effect of culture on choosing the majority had "
        f"LR statistic {lr_culture_majority:.2f} (df = {df_culture_majority}), "
        f"p = {p_culture_majority:.3f}, and the age coefficient was "
        f"{coef_age_majority:.3f} (odds ratio {np.exp(coef_age_majority):.2f}, "
        f"p = {p_age_majority:.3f})."
    )

    if social_varies or majority_varies:
        explanation_parts.append(
            "Because at least one of the age or culture effects on these outcomes was "
            "statistically significant at the 0.05 level, I interpret the data as "
            "providing evidence that children's reliance on social information and/or "
            "their preference for majority cues vary across cultures and developmental "
            "stages. The response value of 75 reflects a relatively strong 'Yes' answer."
        )
    else:
        explanation_parts.append(
            "Neither culture nor age showed statistically significant effects on these "
            "outcomes at the 0.05 level. I therefore interpret the data as lacking strong "
            "evidence that reliance on social information or majority preference varies "
            "systematically across cultures or developmental stages, and the response "
            "value of 25 reflects a fairly confident 'No' answer."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")

    # Also return results in case they are inspected interactively (not required by spec).
    return answer, response, results


if __name__ == "__main__":
    run_analysis()

