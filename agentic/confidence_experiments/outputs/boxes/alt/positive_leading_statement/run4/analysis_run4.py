import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def lr_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    from scipy import stats

    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Create derived variables for social information use and majority preference
    df["social_use"] = (df["y"] != 1).astype(int)
    df_social = df[df["social_use"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Center age to improve numerical stability
    df["age_c"] = df["age"] - df["age"].mean()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # --- Model 1: Reliance on social information (any demonstrated option vs undemonstrated) ---
    m1_full = fit_logit("social_use ~ age_c + C(culture)", df)
    m1_no_age = fit_logit("social_use ~ C(culture)", df)
    m1_no_cult = fit_logit("social_use ~ age_c", df)

    lr_age_social = lr_test(m1_full, m1_no_age)
    lr_cult_social = lr_test(m1_full, m1_no_cult)

    # --- Model 2: Preference for majority vs minority among social users ---
    m2_full = fit_logit("majority_choice ~ age_c + C(culture)", df_social)
    m2_no_age = fit_logit("majority_choice ~ C(culture)", df_social)
    m2_no_cult = fit_logit("majority_choice ~ age_c", df_social)

    lr_age_major = lr_test(m2_full, m2_no_age)
    lr_cult_major = lr_test(m2_full, m2_no_cult)

    # Extract some effect sizes (odds ratios for age)
    age_or_social = float(np.exp(m1_full.params["age_c"]))
    age_or_major = float(np.exp(m2_full.params["age_c"]))

    # Summarize significance
    alpha = 0.05
    age_social_sig = lr_age_social[2] < alpha
    cult_social_sig = lr_cult_social[2] < alpha
    age_major_sig = lr_age_major[2] < alpha
    cult_major_sig = lr_cult_major[2] < alpha

    # Compute log-likelihood pseudo-R2 as a rough measure
    def pseudo_r2(model):
        return 1 - model.llf / model.llnull

    r2_social = float(pseudo_r2(m1_full))
    r2_major = float(pseudo_r2(m2_full))

    # Determine Likert response strength
    # Start from neutral and adjust up/down based on evidence for variation.
    yes_strength = 50

    # Age and culture effects on social information use
    if age_social_sig:
        yes_strength += 10
    else:
        yes_strength -= 5
    if cult_social_sig:
        yes_strength += 10
    else:
        yes_strength -= 5

    # Age and culture effects on majority preference
    if age_major_sig:
        yes_strength += 10
    else:
        yes_strength -= 5
    if cult_major_sig:
        yes_strength += 10
    else:
        yes_strength -= 5

    # Adjust based on overall model fit
    avg_r2 = (r2_social + r2_major) / 2
    if avg_r2 > 0.05:
        yes_strength += 5
    elif avg_r2 < 0.02:
        yes_strength -= 5

    # Clamp to [0, 100]
    response = int(min(max(round(yes_strength), 0), 100))

    any_sig = any([age_social_sig, cult_social_sig, age_major_sig, cult_major_sig])
    if any_sig:
        substantive_lines = [
            "Substantive interpretation:",
            "- At least some age or culture terms reach conventional significance, indicating that reliance on social information and/or majority preference are not constant across development or cultures.",
            "- This suggests that children’s tendency to use social information and to follow majority demonstrations changes with age and differs between at least some cultural sites.",
        ]
    else:
        substantive_lines = [
            "Substantive interpretation:",
            "- None of the age or culture terms reach conventional significance at α = 0.05, and effect sizes plus pseudo-R² values are small.",
            "- Thus, this dataset does not provide clear evidence that reliance on social information or majority preference systematically vary across development or cultures; any observed differences could plausibly reflect sampling variability.",
        ]

    if response >= 60:
        conclusion_line = (
            f"Conclusion on the Likert scale (0 = strong 'No', 100 = strong 'Yes'): "
            f"I assign a response of {response}, reflecting a 'Yes' answer: there is reasonably strong statistical evidence that children’s reliance on social information and their preference for majority cues vary across cultures and developmental stages in this dataset."
        )
    elif response <= 40:
        conclusion_line = (
            f"Conclusion on the Likert scale (0 = strong 'No', 100 = strong 'Yes'): "
            f"I assign a response of {response}, reflecting a 'No' answer: the data do not provide clear statistical evidence that children’s reliance on social information or their preference for majority cues vary systematically across cultures and developmental stages in this dataset."
        )
    else:
        conclusion_line = (
            f"Conclusion on the Likert scale (0 = strong 'No', 100 = strong 'Yes'): "
            f"I assign a response of {response}, reflecting an essentially inconclusive answer: the evidence for systematic variation in children’s reliance on social information and majority preference across cultures and developmental stages is weak and statistically uncertain in this dataset."
        )

    explanation = [
        "Research question: Do children’s reliance on social information and preference for majority cues vary across cultures and developmental stages?",
        "",
        "Data and derived outcomes:",
        "- Outcome y has three categories (1=undemonstrated option, 2=majority option, 3=minority option) across 629 children aged 4–14 from 8 cultural sites.",
        "- I defined social-information use as choosing any demonstrated option (y in {2,3}) versus the undemonstrated option (y=1).",
        "- Among children who used social information (y in {2,3}), I defined majority preference as choosing the majority option (y=2) versus the minority option (y=3).",
        "",
        "Modeling strategy:",
        "- Fitted logistic regression for social-information use: social_use ~ age (centered) + culture (8-level factor).",
        "- Fitted logistic regression for majority preference among social users: majority_choice ~ age (centered) + culture.",
        "- For each model, compared the full model to versions dropping age or culture using likelihood-ratio tests to assess whether age and culture significantly improve model fit.",
        "",
        "Key statistical findings:",
        f"- Social-information use: likelihood-ratio test for age (full vs no-age) yielded p={lr_age_social[2]:.4f}, indicating {'a significant' if age_social_sig else 'no clear'} developmental effect on whether children rely on social information.",
        f"- Social-information use: likelihood-ratio test for culture (full vs no-culture) yielded p={lr_cult_social[2]:.4f}, indicating {'significant' if cult_social_sig else 'no clear'} cross-cultural differences in reliance on social information.",
        f"- Majority preference: likelihood-ratio test for age (full vs no-age) yielded p={lr_age_major[2]:.4f}, indicating {'a significant' if age_major_sig else 'no clear'} developmental effect on majority preference among children who use social information.",
        f"- Majority preference: likelihood-ratio test for culture (full vs no-culture) yielded p={lr_cult_major[2]:.4f}, indicating {'significant' if cult_major_sig else 'no clear'} cross-cultural variation in majority preference.",
        f"- The odds ratio for age in the social-use model is approximately {age_or_social:.2f}, and in the majority-preference model approximately {age_or_major:.2f}, meaning that older children tend to be {'more' if age_or_social > 1 else 'less'} likely to rely on social information and {'more' if age_or_major > 1 else 'less'} likely to follow the majority, per additional year of age.",
        f"- McFadden-style pseudo-R² is about {r2_social:.3f} for social-information use and {r2_major:.3f} for majority preference, suggesting {'modest to moderate' if avg_r2 > 0.05 else 'modest'} explanatory power of age and culture together.",
        "",
        *substantive_lines,
        "",
        conclusion_line,
    ]

    conclusion = {
        "response": response,
        "explanation": "\n".join(explanation),
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
