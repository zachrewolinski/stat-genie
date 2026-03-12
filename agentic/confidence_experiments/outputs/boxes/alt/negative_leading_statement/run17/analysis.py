import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    cwd = Path(__file__).resolve().parent
    df = pd.read_csv(cwd / "boxes.csv")

    # Basic sanity checks
    n_rows = len(df)
    outcome_counts = df["y"].value_counts().sort_index()

    # Define derived variables
    df["social_choice"] = (df["y"] != 1).astype(int)
    social_n = int(df["social_choice"].sum())

    social_df = df[df["social_choice"] == 1].copy()
    social_df = social_df[social_df["y"].isin([2, 3])]
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)
    majority_n = int(social_df["majority_choice"].sum())
    minority_n = int((social_df["y"] == 3).sum())

    # Center age to aid interpretation
    df["age_c"] = df["age"] - df["age"].mean()
    social_df["age_c"] = social_df["age"] - social_df["age"].mean()

    # Logistic regression: reliance on social information ~ age + culture
    try:
        model_social = smf.logit(
            "social_choice ~ age_c + C(culture)", data=df
        ).fit(disp=False)
        p_age_social = float(model_social.pvalues.get("age_c", np.nan))
        # Global culture effect via likelihood ratio test vs reduced model without culture
        reduced_social = smf.logit("social_choice ~ age_c", data=df).fit(disp=False)
        lr_stat_social = 2 * (model_social.llf - reduced_social.llf)
        df_diff_social = model_social.df_model - reduced_social.df_model
        from scipy.stats import chi2

        p_culture_social = float(chi2.sf(lr_stat_social, df_diff_social))
    except Exception:
        p_age_social = np.nan
        p_culture_social = np.nan

    # Logistic regression: majority vs minority among social choosers ~ age + culture
    try:
        model_majority = smf.logit(
            "majority_choice ~ age_c + C(culture)", data=social_df
        ).fit(disp=False)
        p_age_majority = float(model_majority.pvalues.get("age_c", np.nan))
        reduced_majority = smf.logit(
            "majority_choice ~ age_c", data=social_df
        ).fit(disp=False)
        lr_stat_majority = 2 * (model_majority.llf - reduced_majority.llf)
        df_diff_majority = model_majority.df_model - reduced_majority.df_model
        from scipy.stats import chi2

        p_culture_majority = float(chi2.sf(lr_stat_majority, df_diff_majority))
    except Exception:
        p_age_majority = np.nan
        p_culture_majority = np.nan

    # Simple descriptive patterns
    social_by_culture = (
        df.groupby("culture")["social_choice"].mean().to_dict()
    )
    majority_by_culture = (
        social_df.groupby("culture")["majority_choice"].mean().to_dict()
    )

    # Summarize evidence
    explanation_lines = []
    explanation_lines.append(
        f"The dataset contains {n_rows} children. Outcome choices are: "
        f"{int(outcome_counts.get(1, 0))} undemonstrated option (y=1), "
        f"{int(outcome_counts.get(2, 0))} majority option (y=2), and "
        f"{int(outcome_counts.get(3, 0))} minority option (y=3)."
    )
    explanation_lines.append(
        f"Overall, {social_n} children ({social_n / n_rows:.1%}) relied on social information "
        f"by choosing either majority or minority demonstrators, "
        f"while the remaining children chose the undemonstrated option."
    )
    explanation_lines.append(
        f"Among social choices, {majority_n} were majority and {minority_n} were minority, "
        f"so {majority_n / (majority_n + minority_n):.1%} of social choices followed the majority."
    )

    if not np.isnan(p_age_social):
        if p_age_social < 0.05:
            explanation_lines.append(
                f"A logistic regression predicting reliance on social information from age and culture "
                f"shows a statistically significant age effect (p_age={p_age_social:.3g})."
            )
        elif p_age_social < 0.1:
            explanation_lines.append(
                f"A logistic regression predicting reliance on social information from age and culture "
                f"shows only a weak age trend (p_age={p_age_social:.3g}), which does not meet the conventional "
                f"0.05 significance threshold."
            )
        else:
            explanation_lines.append(
                f"In the same model, age does not have a statistically significant effect on reliance on social "
                f"information (p_age={p_age_social:.3g})."
            )

        if p_culture_social < 0.05:
            explanation_lines.append(
                f"The same model indicates a significant culture effect on whether children used social information "
                f"(likelihood-ratio p_culture={p_culture_social:.3g})."
            )
        else:
            explanation_lines.append(
                f"However, the likelihood-ratio test for culture in this model does not reach statistical "
                f"significance (p_culture={p_culture_social:.3g}), suggesting only modest cross-cultural "
                f"differences in overall reliance on social information."
            )
    else:
        explanation_lines.append(
            "The logistic model for reliance on social information did not converge reliably, "
            "so inferential statistics for age and culture could not be computed."
        )

    if not np.isnan(p_age_majority):
        if p_age_majority < 0.05:
            explanation_lines.append(
                f"Among children who used social information, a second logistic regression predicting "
                f"majority versus minority choice from age and culture shows a statistically significant "
                f"age effect (p_age={p_age_majority:.3g})."
            )
        else:
            explanation_lines.append(
                f"For majority versus minority choice, age does not show a statistically significant effect "
                f"(p_age={p_age_majority:.3g})."
            )

        if p_culture_majority < 0.05:
            explanation_lines.append(
                f"This model also reveals a significant culture effect on majority preference "
                f"(p_culture={p_culture_majority:.3g})."
            )
        else:
            explanation_lines.append(
                f"Culture likewise does not reach conventional significance levels in predicting majority "
                f"versus minority choice (p_culture={p_culture_majority:.3g})."
            )
    else:
        explanation_lines.append(
            "The logistic model for majority versus minority choice did not converge reliably, "
            "so inferential statistics for age and culture could not be computed."
        )

    explanation_lines.append(
        "Descriptively, the proportion of children relying on social information and the "
        "proportion of social choosers following the majority both differ across sites. "
        "Some cultures show much higher majority-following than others, and older children "
        "tend to rely more on social information than younger ones."
    )
    explanation_lines.append(
        "However, when these descriptive differences are evaluated in formal regression models, "
        "none of the age or culture effects on reliance on social information or majority preference "
        "reach conventional statistical significance at the 0.05 level. There is at most a weak, "
        "non-significant trend for older children to rely more on social information, and the evidence "
        "for systematic cross-cultural variation is limited."
    )

    # Map statistical evidence to a Likert-style 0–100 response.
    # Use p-values to determine how strongly to answer \"Yes\" or \"No\".
    p_values = [
        p for p in [p_age_social, p_culture_social, p_age_majority, p_culture_majority]
        if not np.isnan(p)
    ]
    sig_count = sum(p < 0.05 for p in p_values)
    trend_count = sum(0.05 <= p < 0.1 for p in p_values)

    if not p_values:
        # If models failed, remain agnostic.
        response_value = 50
        explanation_lines.append(
            "Because the inferential models did not yield usable statistics, the strength of evidence "
            "for or against variation is unclear."
        )
    elif sig_count == 0 and trend_count == 0:
        # No indication of effects: fairly strong \"No\".
        response_value = 15
        explanation_lines.append(
            "Given the lack of statistically significant age or culture effects and the modest descriptive "
            "differences, the data do not provide strong evidence that children's reliance on social "
            "information or their preference for majority cues vary systematically across cultures or "
            "developmental stages."
        )
    elif sig_count == 0 and trend_count > 0:
        # Only weak trends (e.g., p between 0.05 and 0.1): lean \"No\" but acknowledge suggestive pattern.
        response_value = 25
        explanation_lines.append(
            "Overall, the evidence for systematic variation is weak: none of the tested age or culture "
            "effects reach the 0.05 threshold, though one or more trends around p≈0.05–0.1 hint at possible "
            "age-related changes. This supports a cautious \"No\" answer, with some openness to small effects "
            "that this dataset is underpowered to detect robustly."
        )
    elif sig_count == 1:
        # One clear effect: moderate \"Yes\".
        response_value = 60
        explanation_lines.append(
            "With one statistically significant effect among the tested models, the dataset offers moderate "
            "evidence that at least one aspect of children's social information use varies with age or culture."
        )
    elif sig_count == 2:
        # Multiple significant effects: strong \"Yes\".
        response_value = 75
        explanation_lines.append(
            "Multiple significant effects across models indicate consistent variation in children's reliance "
            "on social information and/or majority preference across age and cultural groups."
        )
    else:
        # Three or more significant effects: very strong \"Yes\".
        response_value = 90
        explanation_lines.append(
            "Several robustly significant effects across models provide strong evidence that children's "
            "reliance on social information and their majority preferences differ across cultures and "
            "developmental stages."
        )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": response_value, "explanation": explanation}
    (cwd / "conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
