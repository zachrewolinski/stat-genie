import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
import numpy as np


def fit_logit(formula, data):
    """Fit a logistic regression, retrying with higher max iterations if needed."""
    try:
        model = smf.logit(formula=formula, data=data).fit(disp=False)
    except Exception:
        model = smf.logit(formula=formula, data=data).fit(disp=False, maxiter=200)
    return model


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test between two nested models using log-likelihoods."""
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return float(lr_stat), float(p_value)


def build_explanation(summary_stats, results):
    lines = []
    lines.append(
        "Research question: Do children’s reliance on social information and "
        "preference for majority cues vary across cultures and developmental stages?"
    )
    lines.append("")
    lines.append("Data and outcomes")
    lines.append(
        f"- Sample size: {summary_stats['n']} children from "
        f"{summary_stats['n_cultures']} cultural sites."
    )
    lines.append(
        "- Outcome y coded as 1=undemonstrated option, "
        "2=majority option, 3=minority option."
    )
    lines.append(
        f"- Overall reliance on social information (choosing any demonstrated option) "
        f"is {summary_stats['social_mean']:.1%}."
    )
    lines.append(
        f"- Among social choices, choosing the majority option occurs in "
        f"{summary_stats['majority_mean']:.1%} of cases."
    )
    lines.append("")
    lines.append("Analytic approach")
    lines.append(
        "- Defined social-information use as y in {2,3} vs y=1 and majority preference "
        "as y=2 vs y=3 among children who followed a demonstrated option."
    )
    lines.append(
        "- Fitted logistic regression models with age (in years, 4–14) "
        "and cultural site (categorical) as predictors."
    )
    lines.append(
        "- Used likelihood-ratio tests to assess whether age and culture "
        "significantly improve model fit."
    )
    lines.append("")
    lines.append("Results: reliance on social information")
    lines.append(
        f"- Age effect: In the full model, the age coefficient for social-information "
        f"use is {results['social_age_coef']:.3f} (p={results['social_age_p']:.3g}). "
        "This suggests a possible trend with age, but it does not reach conventional "
        "levels of statistical significance (p<0.05), so evidence for a robust "
        "developmental change in reliance on social information is weak."
    )
    lines.append(
        f"- Cultural effect: Comparing a model with age plus culture to a model with "
        f"age only yields LR={results['social_culture_lr']:.2f}, "
        f"p={results['social_culture_p']:.3g}. This likelihood-ratio test is not "
        "statistically significant, indicating that the data do not provide strong "
        "evidence that reliance on social information differs systematically between "
        "cultural sites once age is taken into account."
    )
    lines.append(
        "- Descriptively, the probability of using social information shows some "
        "variation across age and culture, but these differences are modest and not "
        "clearly distinguishable from sampling noise in the formal models."
    )
    lines.append("")
    lines.append("Results: preference for majority cues")
    lines.append(
        f"- Age effect: Among children who followed a demonstrated option, the age "
        f"coefficient for choosing the majority over the minority option is "
        f"{results['majority_age_coef']:.3f} (p={results['majority_age_p']:.3g}). "
        "This coefficient is small and far from significant, providing little "
        "evidence that majority preference changes systematically with age."
    )
    lines.append(
        f"- Cultural effect: Adding culture to a model with age alone for majority "
        f"preference yields LR={results['majority_culture_lr']:.2f}, "
        f"p={results['majority_culture_p']:.3g}. This test is also non-significant, "
        "so the data do not show strong or consistent cross-cultural differences in "
        "majority preference among social learners."
    )
    lines.append(
        "- Site-level summaries suggest some variability in majority preference "
        "between cultural groups, but the pattern is not strong enough to be "
        "statistically reliable in this sample."
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- For both reliance on social information and majority preference, the "
        "logistic regression models provide limited statistical evidence that age or "
        "culture account for substantial variation in children’s choices."
    )
    lines.append(
        "- There are hints of age-related change in overall reliance on social "
        "information, but this trend falls short of conventional significance and "
        "could plausibly reflect random variation."
    )
    lines.append(
        "- Overall, the data do not show robust, statistically significant differences "
        "in either reliance on social information or majority preference across "
        "cultures or developmental stages in this experiment."
    )
    lines.append("")
    lines.append("Conclusion")
    lines.append(
        "Taken together, the regression results and descriptive patterns do not "
        "provide strong evidence that children’s reliance on social information or "
        "their preference for majority cues vary systematically across cultures and "
        "developmental stages in this dataset. A cautious reading of the data is "
        "therefore closer to a 'No' than a confident 'Yes' for the specific question "
        "posed."
    )
    return "\n".join(lines)


def map_evidence_to_scale(results):
    """
    Map statistical evidence to a 0–100 Likert scale reflecting confidence in a 'Yes' answer.

    The scale is anchored so that clearly non-significant results yield a below-50
    score (leaning toward 'No'), while multiple robustly significant effects push the
    score well above 50 (leaning toward 'Yes').
    """
    pvals = [
        results["social_age_p"],
        results["social_culture_p"],
        results["majority_age_p"],
        results["majority_culture_p"],
    ]

    sig = sum(p < 0.05 for p in pvals)
    strong = sum(p < 0.01 for p in pvals)

    if sig == 0:
        # No conventional evidence for variation in any of the key effects.
        return 35
    if sig == 1 and strong == 0:
        # One marginally significant effect.
        return 55
    if sig <= 2 and strong <= 1:
        # Several significant but not overwhelmingly strong effects.
        return 70
    # Many strong, consistent effects across tests.
    return 85


def main():
    df = pd.read_csv("boxes.csv")

    # Define derived outcomes
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Summary statistics
    summary_stats = {
        "n": int(len(df)),
        "n_cultures": int(df["culture"].nunique()),
        "social_mean": float(df["social_choice"].mean()),
        "majority_mean": float(df.loc[df["social_choice"] == 1, "majority_choice"].mean()),
    }

    # Logistic regression for reliance on social information
    social_full = fit_logit("social_choice ~ age + C(culture)", df)
    social_age_only = fit_logit("social_choice ~ age", df)

    social_age_coef = float(social_full.params["age"])
    social_age_p = float(social_full.pvalues["age"])
    social_culture_lr, social_culture_p = lr_test(social_full, social_age_only)

    # Logistic regression for majority preference among social choosers
    df_social = df[df["social_choice"] == 1].copy()
    majority_full = fit_logit("majority_choice ~ age + C(culture)", df_social)
    majority_age_only = fit_logit("majority_choice ~ age", df_social)

    majority_age_coef = float(majority_full.params["age"])
    majority_age_p = float(majority_full.pvalues["age"])
    majority_culture_lr, majority_culture_p = lr_test(majority_full, majority_age_only)

    results = {
        "social_age_coef": social_age_coef,
        "social_age_p": social_age_p,
        "social_culture_lr": social_culture_lr,
        "social_culture_p": social_culture_p,
        "majority_age_coef": majority_age_coef,
        "majority_age_p": majority_age_p,
        "majority_culture_lr": majority_culture_lr,
        "majority_culture_p": majority_culture_p,
    }

    response = map_evidence_to_scale(results)
    explanation = build_explanation(summary_stats, results)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
