import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Define derived variables
    # social: 1 if child chose a demonstrated option (majority or minority), 0 if undemonstrated option
    df["social"] = df["majority_first"].isin([2, 3]).astype(int)

    # majority_choice: among social learners, 1 if chose majority option, 0 if chose minority
    df["majority_choice"] = np.where(
        df["majority_first"] == 2,
        1,
        np.where(df["majority_first"] == 3, 0, np.nan),
    )

    # Use site ID as proxy for cultural context
    df["site"] = df["y"].astype(int)

    # Basic descriptive stats
    n = len(df)
    social_rate = df["social"].mean()
    imitators = df[df["social"] == 1]
    majority_rate = imitators["majority_choice"].mean()

    # Age-grouped summaries to capture developmental stages
    bins = [4, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    age_group_summary = (
        df.groupby("age_group")
        .agg(
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
            n=("social", "size"),
        )
        .dropna(subset=["n"])
    )

    # --- Tests for variation in social-information use ---
    # Site (culture) differences: chi-square test of independence
    social_site_table = pd.crosstab(df["site"], df["social"])
    chi2_social_site, p_social_site, _, _ = stats.chi2_contingency(social_site_table)

    # Age effect: logistic regression with continuous age
    social_model = smf.logit("social ~ age", data=df).fit(disp=False)
    age_coef_social = social_model.params["age"]
    age_p_social = social_model.pvalues["age"]

    # Predicted change in social use from age 4 to 14
    def logistic(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    intercept_social = social_model.params["Intercept"]
    logit_4 = intercept_social + age_coef_social * 4
    logit_14 = intercept_social + age_coef_social * 14
    prob_social_4 = logistic(logit_4)
    prob_social_14 = logistic(logit_14)
    delta_social_age = prob_social_14 - prob_social_4

    # --- Tests for variation in majority preference among social learners ---
    # Work only with children who followed some demonstrator
    imitators = df[df["social"] == 1].copy()

    # Guard against degenerate cases, though dataset should have both outcomes
    if imitators["majority_choice"].nunique() > 1:
        majority_site_table = pd.crosstab(imitators["site"], imitators["majority_choice"])
        chi2_majority_site, p_majority_site, _, _ = stats.chi2_contingency(
            majority_site_table
        )

        majority_model = smf.logit("majority_choice ~ age", data=imitators).fit(
            disp=False
        )
        age_coef_majority = majority_model.params["age"]
        age_p_majority = majority_model.pvalues["age"]

        intercept_majority = majority_model.params["Intercept"]
        logit_4_m = intercept_majority + age_coef_majority * 4
        logit_14_m = intercept_majority + age_coef_majority * 14
        prob_majority_4 = logistic(logit_4_m)
        prob_majority_14 = logistic(logit_14_m)
        delta_majority_age = prob_majority_14 - prob_majority_4
    else:
        # Fallback: no variation to model
        chi2_majority_site = np.nan
        p_majority_site = 1.0
        age_coef_majority = 0.0
        age_p_majority = 1.0
        prob_majority_4 = prob_majority_14 = imitators["majority_choice"].iloc[0]
        delta_majority_age = 0.0

    # --- Synthesize evidence into a Likert-style response ---
    evidence_bits = []

    # Social-information use variation
    if p_social_site < 0.05:
        evidence_bits.append(
            f"Social-information use (choosing any demonstrated option) varies significantly across sites (chi2={chi2_social_site:.2f}, p={p_social_site:.3f})."
        )
    else:
        evidence_bits.append(
            f"Social-information use shows no strong overall site effect (chi2={chi2_social_site:.2f}, p={p_social_site:.3f})."
        )

    if age_p_social < 0.05:
        direction = "increases" if age_coef_social > 0 else "decreases"
        evidence_bits.append(
            f"The probability of using social information {direction} with age; the model predicts a change of {delta_social_age:.2f} (from age 4 to 14)."
        )
    else:
        evidence_bits.append(
            "There is no statistically robust linear age trend in overall social-information use."
        )

    # Majority preference variation
    if p_majority_site < 0.05:
        evidence_bits.append(
            f"Among social learners, majority preference differs significantly across sites (chi2={chi2_majority_site:.2f}, p={p_majority_site:.3f})."
        )
    else:
        evidence_bits.append(
            f"Among social learners, majority preference does not show a strong overall site effect (chi2={chi2_majority_site:.2f}, p={p_majority_site:.3f})."
        )

    if age_p_majority < 0.05:
        direction_m = "increases" if age_coef_majority > 0 else "decreases"
        evidence_bits.append(
            f"Conditional on copying, the probability of choosing the majority option {direction_m} with age; the model predicts a change of {delta_majority_age:.2f} (from age 4 to 14)."
        )
    else:
        evidence_bits.append(
            "Conditional on copying, majority preference does not show a clear linear trend with age."
        )

    # Age-group summaries for interpretability
    age_group_lines = []
    for idx, row in age_group_summary.iterrows():
        age_group_lines.append(
            f"Age {idx}: social-use rate={row['social_rate']:.2f}, majority preference among social learners={row['majority_rate']:.2f} (n={int(row['n'])})."
        )

    # Decide on overall Likert-scale response (0-100, higher = stronger YES)
    # Start from neutral and adjust based on significance and effect sizes.
    score = 50

    # Social-information variation
    if p_social_site < 0.05:
        score += 10
    if abs(delta_social_age) >= 0.10 and age_p_social < 0.05:
        score += 10

    # Majority-preference variation
    if p_majority_site < 0.05:
        score += 10
    if abs(delta_majority_age) >= 0.10 and age_p_majority < 0.05:
        score += 10

    # Clamp to [0, 100]
    score = int(min(100, max(0, round(score))))

    # Build explanation text (single line, no newlines)
    explanation_parts = [
        f"Overall, {n} children were tested across eight sites, with {social_rate:.2f} using social information (choosing a demonstrated option) and, among those, {majority_rate:.2f} choosing the majority option.",
        "We treat site ID as a proxy for cultural context and age (4–14 years) as developmental stage.",
    ]
    explanation_parts.extend(evidence_bits)
    explanation_parts.append(
        "Age-group summaries show the following pattern: " + " ".join(age_group_lines)
    )
    explanation_parts.append(
        "Taken together, these patterns provide "
        + ("strong" if score >= 70 else "moderate" if score >= 60 else "limited")
        + " evidence that both reliance on social information and preference for majority cues vary across cultures and developmental stages in this dataset."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

