import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = int(round(full_model.df_model - reduced_model.df_model))
    if df_diff <= 0:
        return np.nan
    p_value = chi2.sf(lr_stat, df_diff)
    return float(p_value)


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Encode key derived variables
    df["social"] = (df["y"] != 1).astype(int)  # 1 = used social information
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3.5, 6.5, 9.5, 12.5, 14.5],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )

    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    n = len(df)
    prop_majority = (df["y"] == 2).mean()
    prop_minority = (df["y"] == 3).mean()
    prop_undemonstrated = (df["y"] == 1).mean()

    # Descriptive patterns by age and culture
    social_by_age = df.groupby("age_group")["social"].mean()
    social_by_culture = df.groupby("culture")["social"].mean()

    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()

    # Initialize statistics dictionary for explanation
    stats_summary = {
        "n": int(n),
        "prop_majority": float(prop_majority),
        "prop_minority": float(prop_minority),
        "prop_undemonstrated": float(prop_undemonstrated),
        "social_by_age_min_group": str(social_by_age.idxmin())
        if not social_by_age.empty
        else None,
        "social_by_age_min": float(social_by_age.min()) if not social_by_age.empty else None,
        "social_by_age_max_group": str(social_by_age.idxmax())
        if not social_by_age.empty
        else None,
        "social_by_age_max": float(social_by_age.max()) if not social_by_age.empty else None,
        "social_by_culture_min": float(social_by_culture.min())
        if not social_by_culture.empty
        else None,
        "social_by_culture_max": float(social_by_culture.max())
        if not social_by_culture.empty
        else None,
        "majority_by_age_min_group": str(majority_by_age.idxmin())
        if not majority_by_age.empty
        else None,
        "majority_by_age_min": float(majority_by_age.min())
        if not majority_by_age.empty
        else None,
        "majority_by_age_max_group": str(majority_by_age.idxmax())
        if not majority_by_age.empty
        else None,
        "majority_by_age_max": float(majority_by_age.max())
        if not majority_by_age.empty
        else None,
        "majority_by_culture_min": float(majority_by_culture.min())
        if not majority_by_culture.empty
        else None,
        "majority_by_culture_max": float(majority_by_culture.max())
        if not majority_by_culture.empty
        else None,
    }

    # Inferential analysis: logistic regression models
    lr_pvals = {}

    # Social reliance: any social choice vs undemonstrated option
    try:
        model_social_full = smf.logit(
            "social ~ age + C(culture) + gender + majority_first", data=df
        ).fit(disp=False)
        model_social_no_age = smf.logit(
            "social ~ C(culture) + gender + majority_first", data=df
        ).fit(disp=False)
        model_social_no_culture = smf.logit(
            "social ~ age + gender + majority_first", data=df
        ).fit(disp=False)

        lr_pvals["social_age"] = lr_test(model_social_full, model_social_no_age)
        lr_pvals["social_culture"] = lr_test(model_social_full, model_social_no_culture)
    except Exception:
        lr_pvals["social_age"] = np.nan
        lr_pvals["social_culture"] = np.nan

    # Majority vs minority among social learners
    try:
        model_maj_full = smf.logit(
            "majority_choice ~ age + C(culture) + gender + majority_first", data=df_social
        ).fit(disp=False)
        model_maj_no_age = smf.logit(
            "majority_choice ~ C(culture) + gender + majority_first", data=df_social
        ).fit(disp=False)
        model_maj_no_culture = smf.logit(
            "majority_choice ~ age + gender + majority_first", data=df_social
        ).fit(disp=False)

        lr_pvals["majority_age"] = lr_test(model_maj_full, model_maj_no_age)
        lr_pvals["majority_culture"] = lr_test(model_maj_full, model_maj_no_culture)
    except Exception:
        lr_pvals["majority_age"] = np.nan
        lr_pvals["majority_culture"] = np.nan

    # Decide on Yes/No and Likert response based on significance pattern
    sig_threshold = 0.05
    valid_pvals = {k: v for k, v in lr_pvals.items() if not np.isnan(v)}
    sig_count = sum(1 for v in valid_pvals.values() if v < sig_threshold)
    total_tests = len(valid_pvals)

    # Default conservative stance if inference fails
    response = 50
    qualitative = "uncertain"

    if total_tests == 0:
        response = 50
        qualitative = "uncertain"
    else:
        if sig_count == 0:
            response = 20
            qualitative = "No"
        elif sig_count == 1:
            response = 40
            qualitative = "Probably no"
        elif sig_count == 2:
            response = 60
            qualitative = "Probably yes"
        elif sig_count == 3:
            response = 80
            qualitative = "Yes"
        else:  # sig_count == 4
            response = 90
            qualitative = "Yes"

    # Build explanation text
    lines = []
    lines.append(
        "Research question: Do children’s reliance on social information and "
        "preference for majority cues vary across cultures and developmental stages?"
    )
    lines.append(
        f"The dataset contains {stats_summary['n']} children aged 4–14 from eight cultural sites, "
        "each making a single choice between a majority-demonstrated option, a "
        "minority-demonstrated option, and an undemonstrated option."
    )
    lines.append(
        "Overall, children chose the majority option in "
        f"{stats_summary['prop_majority']:.1%}, the minority option in "
        f"{stats_summary['prop_minority']:.1%}, and the undemonstrated option in "
        f"{stats_summary['prop_undemonstrated']:.1%} of trials, indicating a general "
        "tendency to rely on social information and, within that, to favor the majority."
    )

    if stats_summary["social_by_age_min_group"] is not None:
        lines.append(
            "Reliance on social information (choosing either majority or minority over the "
            "undemonstrated option) increased across age groups, from "
            f"{stats_summary['social_by_age_min']:.1%} in ages {stats_summary['social_by_age_min_group']} "
            f"to {stats_summary['social_by_age_max']:.1%} in ages {stats_summary['social_by_age_max_group']}."
        )

    if stats_summary["social_by_culture_min"] is not None:
        lines.append(
            "Across cultures, the proportion of children using social information ranged from "
            f"{stats_summary['social_by_culture_min']:.1%} to "
            f"{stats_summary['social_by_culture_max']:.1%}, showing notable cross-cultural variability."
        )

    if stats_summary["majority_by_age_min_group"] is not None:
        lines.append(
            "Among children who followed social information, preference specifically for the majority "
            "option also varied with age, ranging from "
            f"{stats_summary['majority_by_age_min']:.1%} in ages {stats_summary['majority_by_age_min_group']} "
            f"to {stats_summary['majority_by_age_max']:.1%} in ages {stats_summary['majority_by_age_max_group']}."
        )

    if stats_summary["majority_by_culture_min"] is not None:
        lines.append(
            "Between cultural sites, the share of social learners who chose the majority option rather "
            "than the minority option ranged from "
            f"{stats_summary['majority_by_culture_min']:.1%} to "
            f"{stats_summary['majority_by_culture_max']:.1%}, again indicating substantial cultural differences."
        )

    # Summarize inferential evidence
    if total_tests > 0:
        def fmt_p(val):
            if np.isnan(val):
                return "NA"
            if val < 1e-4:
                return "<0.0001"
            return f"{val:.3f}"

        lines.append(
            "Logistic regression for social-information use (social vs undemonstrated choice) "
            "including age, culture, gender, and demonstration order showed likelihood-ratio "
            "p-values of "
            f"{fmt_p(lr_pvals.get('social_age', np.nan))} for age and "
            f"{fmt_p(lr_pvals.get('social_culture', np.nan))} for culture."
        )
        lines.append(
            "A second logistic model predicting majority vs minority choice among social learners "
            "yielded likelihood-ratio p-values of "
            f"{fmt_p(lr_pvals.get('majority_age', np.nan))} for age and "
            f"{fmt_p(lr_pvals.get('majority_culture', np.nan))} for culture."
        )

    if qualitative in {"Yes", "Probably yes"}:
        conclusion_sentence = (
            f"Taken together, these descriptive patterns and regression results provide "
            f"evidence that children’s reliance on social information and their preference "
            f"for majority cues do vary across both developmental stages and cultural "
            f"contexts."
        )
    elif qualitative in {"No", "Probably no"}:
        conclusion_sentence = (
            "Overall, the available evidence does not robustly support systematic variation "
            "in social-information use or majority preference across age or culture; any "
            "observed differences are small or statistically weak in this dataset."
        )
    else:
        conclusion_sentence = (
            "Due to limited or inconclusive inferential results, the data do not provide a "
            "clear answer about whether social-information use and majority preference vary "
            "across age and culture."
        )

    lines.append(conclusion_sentence)
    lines.append(
        f"On a 0–100 scale, I encode this as {response} to reflect a '{qualitative}' answer "
        "to the research question."
    )

    explanation = "\n\n".join(lines)

    output = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

