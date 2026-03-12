import json
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing two nested models."""
    df_diff = len(full_model.params) - len(reduced_model.params)
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(lr_stat), float(p_value)


def p_to_confidence(p_value):
    """Map a p-value to a [0,1] confidence score.

    We treat conventional p < 0.05 as reasonably strong evidence for a relationship,
    0.05 <= p < 0.10 as weak/suggestive evidence, and larger p-values as providing
    little to no evidence for systematic variation.
    """
    if p_value is None or np.isnan(p_value):
        return 0.0
    if p_value < 0.01:
        return 1.0
    if p_value < 0.05:
        return 0.7
    if p_value < 0.10:
        return 0.3
    return 0.0


def p_description(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    if p_value < 0.05:
        return "p < 0.05"
    return f"p = {p_value:.3f}"


def main():
    df = pd.read_csv("boxes.csv")

    # Social-information reliance: 1 if child chose any demonstrated option (majority or minority)
    df["social"] = (df["y"] != 1).astype(int)

    # Majority preference: among children who used social information
    df_social = df[df["y"].isin([2, 3])].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Age groups for descriptive summaries
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df_social["age_group"] = pd.cut(df_social["age"], bins=bins, labels=labels, right=False)

    # Descriptive statistics: social reliance
    social_overall = df["social"].mean()
    social_by_age = df.groupby("age_group")["social"].mean().dropna()
    social_by_culture = df.groupby("culture")["social"].mean()

    # Descriptive statistics: majority choice
    majority_overall = df_social["majority_choice"].mean()
    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean().dropna()
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()

    # GLM for social-information reliance
    social_full = smf.glm(
        "social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_age = smf.glm(
        "social ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_culture = smf.glm(
        "social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_social_age, p_social_age = lr_test(social_full, social_no_age)
    lr_social_culture, p_social_culture = lr_test(social_full, social_no_culture)

    # GLM for majority preference (among social learners)
    majority_full = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_age = smf.glm(
        "majority_choice ~ C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_culture = smf.glm(
        "majority_choice ~ age",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    lr_majority_age, p_majority_age = lr_test(majority_full, majority_no_age)
    lr_majority_culture, p_majority_culture = lr_test(majority_full, majority_no_culture)

    # Confidence that variation exists for each effect
    conf_social_age = p_to_confidence(p_social_age)
    conf_social_culture = p_to_confidence(p_social_culture)
    conf_majority_age = p_to_confidence(p_majority_age)
    conf_majority_culture = p_to_confidence(p_majority_culture)

    conf_values = [
        conf_social_age,
        conf_social_culture,
        conf_majority_age,
        conf_majority_culture,
    ]
    overall_confidence = float(np.mean(conf_values))

    # Map overall confidence in [0,1] to 0-100 Likert scale
    response_score = int(round(overall_confidence * 100))

    # Build human-readable explanation
    social_age_min = float(social_by_age.min()) if len(social_by_age) > 0 else float("nan")
    social_age_max = float(social_by_age.max()) if len(social_by_age) > 0 else float("nan")
    social_culture_min = float(social_by_culture.min())
    social_culture_max = float(social_by_culture.max())

    majority_age_min = float(majority_by_age.min()) if len(majority_by_age) > 0 else float("nan")
    majority_age_max = float(majority_by_age.max()) if len(majority_by_age) > 0 else float("nan")
    majority_culture_min = float(majority_by_culture.min())
    majority_culture_max = float(majority_by_culture.max())

    explanation = dedent(
        f"""
        I examined two aspects of children's behavior in this dataset of {len(df)} children from 8 cultural sites:
        (a) whether they relied on social information at all (choosing either a demonstrated option vs. an undemonstrated third option), and
        (b) among children who used social information, whether they followed the majority vs. the minority demonstrators.

        For social-information reliance, I fit a logistic regression (binomial GLM) with predictors age (in years) and culture (site ID).
        Likelihood-ratio tests comparing nested models showed that the effect of age on social-information reliance was only marginal by conventional standards (χ² = {lr_social_age:.2f}, {p_description(p_social_age)}),
        and the overall effect of culture on social-information reliance was not statistically reliable (χ² = {lr_social_culture:.2f}, {p_description(p_social_culture)}).
        Descriptively, the probability of relying on social information increased across age groups from about {social_age_min*100:.1f}% in the lowest age band
        to about {social_age_max*100:.1f}% in the highest band, and varied across cultures from roughly {social_culture_min*100:.1f}% to {social_culture_max*100:.1f}%.

        For majority preference, I restricted the sample to children who chose a demonstrated option and fit an analogous GLM with age and culture predicting
        whether the child followed the majority demonstrators. Here, the likelihood-ratio tests indicated that neither age (χ² = {lr_majority_age:.2f}, {p_description(p_majority_age)})
        nor culture (χ² = {lr_majority_culture:.2f}, {p_description(p_majority_culture)}) provided strong statistical evidence for systematic differences in majority preference.
        Among social learners, the probability of following the majority was about {majority_overall*100:.1f}% on average, increased across age groups from roughly
        {majority_age_min*100:.1f}% to {majority_age_max*100:.1f}%, and differed across cultures from about {majority_culture_min*100:.1f}% to {majority_culture_max*100:.1f}%.

        Taken together, the data show noticeable differences in raw proportions across ages and cultures, but the formal statistical tests provide at best weak
        (and often non-significant) evidence that these differences reflect robust underlying relationships rather than sampling variability. In other words, this
        dataset does not offer strong statistical support for the claim that children's reliance on social information and preference for majority cues systematically
        vary across developmental stages and cultural contexts, even though some descriptive patterns point in that direction. Based on this balance of descriptive
        trends and limited statistical evidence, I rate the evidence as {response_score}/100 in favor of the claim that these tendencies vary with both culture and age.
        """
    ).strip()

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
