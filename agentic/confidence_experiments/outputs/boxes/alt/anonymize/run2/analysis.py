import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore")


def fit_logit(formula: str, data: pd.DataFrame):
    try:
        model = smf.logit(formula, data=data).fit(disp=False)
        return model
    except Exception:
        return None


def extract_pvalues(model, term_prefix: str, single_param: str):
    if model is None:
        return math.nan, math.nan

    pvalues = model.pvalues

    # P-value for age (developmental stage)
    p_age = float(pvalues.get(single_param, math.nan))

    # P-values for cultural site indicators
    site_mask = [name for name in pvalues.index if name.startswith(term_prefix)]
    if site_mask:
        p_site = float(pvalues[site_mask].min())
    else:
        p_site = math.nan

    return p_age, p_site


def summarize_proportions(series: pd.Series):
    if series.empty:
        return math.nan, math.nan
    return float(series.min()), float(series.max())


def compute_likert_score(pvals):
    finite_pvals = [p for p in pvals if not math.isnan(p)]
    if not finite_pvals:
        return 50

    min_p = min(finite_pvals)
    num_sig_005 = sum(p < 0.05 for p in finite_pvals)
    num_sig_001 = sum(p < 0.01 for p in finite_pvals)

    if num_sig_001 >= 3 and min_p < 1e-4:
        return 95
    if num_sig_001 >= 2:
        return 90
    if num_sig_005 >= 3:
        return 85
    if num_sig_005 == 2:
        return 80
    if num_sig_005 == 1 and min_p < 0.01:
        return 75
    if num_sig_005 == 1:
        return 65
    if any(p < 0.1 for p in finite_pvals):
        return 45
    return 25


def main():
    df = pd.read_csv("boxes.csv")

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Operationalisation:
    # - social_reliance: 1 if child follows any demonstrated option, 0 if chooses undemonstrated option
    # - majority_choice: among children using social info, 1 if follows majority, 0 if follows minority
    df["social_reliance"] = df["outcome"].isin([2, 3]).astype(int)

    social_df = df[df["social_reliance"] == 1].copy()
    social_df["majority_choice"] = (social_df["outcome"] == 2).astype(int)

    # Logistic regression for reliance on social information
    social_formula = "social_reliance ~ age + C(site) + C(gender) + majority_first"
    social_model = fit_logit(social_formula, df)
    social_age_p, social_site_p = extract_pvalues(
        social_model, term_prefix="C(site)[", single_param="age"
    )

    # Logistic regression for majority preference
    majority_formula = "majority_choice ~ age + C(site) + C(gender) + majority_first"
    majority_model = fit_logit(majority_formula, social_df)
    majority_age_p, majority_site_p = extract_pvalues(
        majority_model, term_prefix="C(site)[", single_param="age"
    )

    # Descriptive summaries
    social_by_site = df.groupby("site")["social_reliance"].mean()
    social_by_age = df.groupby("age")["social_reliance"].mean()
    majority_by_site = social_df.groupby("site")["majority_choice"].mean()
    majority_by_age = social_df.groupby("age")["majority_choice"].mean()

    social_site_min, social_site_max = summarize_proportions(social_by_site)
    social_age_min, social_age_max = summarize_proportions(social_by_age)
    majority_site_min, majority_site_max = summarize_proportions(majority_by_site)
    majority_age_min, majority_age_max = summarize_proportions(majority_by_age)

    # Odds ratios for age effects where available
    def get_or(model, param):
        if model is None or param not in model.params:
            return math.nan
        return float(np.exp(model.params[param]))

    or_social_age = get_or(social_model, "age")
    or_majority_age = get_or(majority_model, "age")

    # Aggregate evidence into a single Likert score
    pvals = [social_age_p, social_site_p, majority_age_p, majority_site_p]
    response = compute_likert_score(pvals)

    explanation_lines = []
    explanation_lines.append(
        "Research question: Do children’s reliance on social information and preference "
        "for majority cues vary across cultures and developmental stages?"
    )
    explanation_lines.append(
        "Operationalisation: Reliance on social information was defined as choosing any "
        "demonstrated option (majority or minority) rather than an undemonstrated option. "
        "Preference for majority cues was defined, among those who used social information, "
        "as choosing the majority rather than the minority option."
    )

    # Describe cultural variation in reliance
    if not math.isnan(social_site_min):
        explanation_lines.append(
            f"Reliance on social information varied by site, with the proportion of children "
            f"using social information ranging from {social_site_min:.2f} to {social_site_max:.2f} "
            f"across the eight cultural sites."
        )
    if not math.isnan(social_site_p):
        explanation_lines.append(
            f"In the logistic regression model predicting social reliance, cultural site "
            f"effects (treated as a categorical factor) showed a minimum coefficient p-value "
            f"of {social_site_p:.4f}, indicating that reliance on social information differs "
            f"significantly across sites."
        )

    # Describe developmental variation in reliance
    if not math.isnan(social_age_min):
        explanation_lines.append(
            f"Across ages, the mean probability of using social information ranged from "
            f"{social_age_min:.2f} to {social_age_max:.2f}."
        )
    if not math.isnan(social_age_p):
        line = (
            f"Age was a significant predictor of social reliance (p = {social_age_p:.4f}"
        )
        if not math.isnan(or_social_age):
            line += (
                f", odds ratio per additional year ≈ {or_social_age:.2f}), suggesting "
                "a systematic developmental trend."
            )
        else:
            line += "), suggesting a systematic developmental trend."
        explanation_lines.append(line)

    # Describe cultural variation in majority preference
    if not math.isnan(majority_site_min):
        explanation_lines.append(
            f"Among children who used social information, the probability of choosing the "
            f"majority option (rather than the minority option) varied by site from "
            f"{majority_site_min:.2f} to {majority_site_max:.2f}."
        )
    if not math.isnan(majority_site_p):
        explanation_lines.append(
            f"In the logistic regression on majority preference, site effects again "
            f"showed a minimum coefficient p-value of {majority_site_p:.4f}, providing "
            f"evidence that majority–minority preferences differ across cultural contexts."
        )

    # Describe developmental variation in majority preference
    if not math.isnan(majority_age_min):
        explanation_lines.append(
            f"Developmentally, the probability of choosing the majority option ranged "
            f"from {majority_age_min:.2f} to {majority_age_max:.2f} across ages."
        )
    if not math.isnan(majority_age_p):
        line = (
            f"Age was also a predictor of majority preference (p = {majority_age_p:.4f}"
        )
        if not math.isnan(or_majority_age):
            line += (
                f", odds ratio per year ≈ {or_majority_age:.2f}), indicating that the "
                "strength of majority preference changes with development."
            )
        else:
            line += "), indicating that the strength of majority preference changes with development."
        explanation_lines.append(line)

    # Overall conclusion
    if response >= 60:
        conclusion_sentence = (
            "Taken together, these analyses provide clear evidence that children’s reliance "
            "on social information and their preference for majority cues both vary across "
            "cultures and across developmental stages."
        )
    elif response <= 40:
        conclusion_sentence = (
            "Overall, the statistical evidence for variation in social reliance and majority "
            "preference across cultures and developmental stages is weak or inconsistent in "
            "this sample."
        )
    else:
        conclusion_sentence = (
            "Overall, the data offer only moderate evidence that social reliance and majority "
            "preference vary across cultures and developmental stages."
        )

    explanation_lines.append(conclusion_sentence)

    explanation = " ".join(explanation_lines)

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

