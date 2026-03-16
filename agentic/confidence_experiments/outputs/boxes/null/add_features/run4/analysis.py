import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(chi2, n, r, k):
    """Compute Cramer's V effect size for contingency tables."""
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Basic sanity checks
    df = df.dropna(subset=["y", "age", "culture"])

    # Outcome coding: 1 = undemonstrated option, 2 = majority, 3 = minority
    df["uses_social"] = df["y"].isin([2, 3]).astype(int)

    # Restrict to trials where child chose a demonstrated option
    demo_df = df[df["y"].isin([2, 3])].copy()
    demo_df["majority_choice"] = (demo_df["y"] == 2).astype(int)

    # Treat culture as categorical and age as ordered categorical (developmental stages)
    df["culture"] = df["culture"].astype(int).astype(str)
    demo_df["culture"] = demo_df["culture"].astype(int).astype(str)

    # Age is coded into discrete bands (e.g., 17.5, 22, ..., 57); treat each unique value as a "stage".
    df["age_stage"] = df["age"].astype(float).astype(str)
    demo_df["age_stage"] = demo_df["age"].astype(float).astype(str)

    results = {}

    # 1. Does reliance on social information (uses_social) vary by culture?
    ct_culture_social = pd.crosstab(df["uses_social"], df["culture"])
    chi2_cult_social, p_cult_social, dof_cs, _ = chi2_contingency(ct_culture_social)
    v_cult_social = cramers_v(chi2_cult_social, df.shape[0], *ct_culture_social.shape)

    results["social_by_culture"] = {
        "chi2": chi2_cult_social,
        "p_value": p_cult_social,
        "dof": dof_cs,
        "cramers_v": v_cult_social,
        "table": ct_culture_social.to_dict(),
    }

    # 2. Does reliance on social information vary by developmental stage (age_stage)?
    ct_age_social = pd.crosstab(df["uses_social"], df["age_stage"])
    chi2_age_social, p_age_social, dof_as, _ = chi2_contingency(ct_age_social)
    v_age_social = cramers_v(chi2_age_social, df.shape[0], *ct_age_social.shape)

    results["social_by_age"] = {
        "chi2": chi2_age_social,
        "p_value": p_age_social,
        "dof": dof_as,
        "cramers_v": v_age_social,
        "table": ct_age_social.to_dict(),
    }

    # 3. Among those who used social info, does preference for majority vs minority vary by culture?
    ct_culture_majority = pd.crosstab(demo_df["majority_choice"], demo_df["culture"])
    chi2_cult_maj, p_cult_maj, dof_cm, _ = chi2_contingency(ct_culture_majority)
    v_cult_maj = cramers_v(chi2_cult_maj, demo_df.shape[0], *ct_culture_majority.shape)

    results["majority_by_culture"] = {
        "chi2": chi2_cult_maj,
        "p_value": p_cult_maj,
        "dof": dof_cm,
        "cramers_v": v_cult_maj,
        "table": ct_culture_majority.to_dict(),
    }

    # 4. Among those who used social info, does preference for majority vs minority vary by age_stage?
    ct_age_majority = pd.crosstab(demo_df["majority_choice"], demo_df["age_stage"])
    chi2_age_maj, p_age_maj, dof_am, _ = chi2_contingency(ct_age_majority)
    v_age_maj = cramers_v(chi2_age_maj, demo_df.shape[0], *ct_age_majority.shape)

    results["majority_by_age"] = {
        "chi2": chi2_age_maj,
        "p_value": p_age_maj,
        "dof": dof_am,
        "cramers_v": v_age_maj,
        "table": ct_age_majority.to_dict(),
    }

    # Also compute overall social reliance and majority preference rates for context.
    overall_social_rate = df["uses_social"].mean()
    overall_majority_rate = demo_df["majority_choice"].mean()

    results["overall"] = {
        "social_rate": overall_social_rate,
        "majority_rate_given_social": overall_majority_rate,
    }

    # Summarize evidence into a single Likert-style response.
    # Heuristic mapping: base on significance (p-values) and effect sizes (Cramer's V).
    pvals = [
        p_cult_social,
        p_age_social,
        p_cult_maj,
        p_age_maj,
    ]
    vs = [
        v_cult_social,
        v_age_social,
        v_cult_maj,
        v_age_maj,
    ]

    n_sig = sum(p < 0.05 for p in pvals)
    n_strong_sig = sum(p < 0.01 for p in pvals)

    # Rough effect size categories (Cramer's V): 0.1 small, 0.3 medium, 0.5 large
    mean_v = float(np.mean(vs))

    # Start from an agnostic midpoint
    score = 50

    # Adjust for number and strength of significant relationships
    if n_sig == 0:
        score = 20
    elif n_sig == 1:
        score = 45
    elif n_sig == 2:
        score = 60
    elif n_sig == 3:
        score = 75
    else:  # all four significant
        score = 85

    # Nudge based on average effect size
    if mean_v >= 0.5:
        score += 10
    elif mean_v >= 0.3:
        score += 5
    elif mean_v <= 0.1:
        score -= 5

    # Clip to [0, 100] and cast to int
    score = int(max(0, min(100, round(score))))

    # Build a human-readable explanation of the evidence
    def fmt_pct(x):
        return round(float(x) * 100, 1)

    explanation_parts = []
    explanation_parts.append(
        "The dataset records whether each child chose the undemonstrated option (1), "
        "the majority option (2), or the minority option (3), along with their age and cultural site."
    )
    explanation_parts.append(
        f"Overall, {fmt_pct(overall_social_rate)}% of choices followed social information "
        f"(majority or minority options), and among those, {fmt_pct(overall_majority_rate)}% "
        "followed the majority rather than the minority."
    )
    explanation_parts.append(
        "To test whether reliance on social information varies across cultures and developmental stages, "
        "I ran chi-square tests on contingency tables of social vs. asocial choices by culture and by age-coded stages."
        f" The culture-by-social test yielded p={p_cult_social:.4g} with Cramer's V={v_cult_social:.3f}, "
        f"and the age-stage-by-social test yielded p={p_age_social:.4g} with Cramer's V={v_age_social:.3f}."
    )
    explanation_parts.append(
        "To test whether preference for majority over minority demonstrations varies, I restricted to trials "
        "where children chose a demonstrated option and compared majority vs. minority choices across culture and age stages."
        f" The culture-by-majority test gave p={p_cult_maj:.4g} with Cramer's V={v_cult_maj:.3f}, "
        f"and the age-stage-by-majority test gave p={p_age_maj:.4g} with Cramer's V={v_age_maj:.3f}."
    )
    explanation_parts.append(
        "Combining the pattern of p-values (number of statistically significant tests) with the effect sizes "
        "(average Cramer's V of "
        f"{mean_v:.3f}), I map the evidence to a single Likert-style score indicating how strongly the data support "
        "the claim that children's reliance on social information and their preference for majority cues vary across "
        "cultures and developmental stages."
    )
    explanation_parts.append(
        f"The resulting score is {score} on a 0–100 scale, where higher values indicate stronger evidence that "
        "both social reliance and majority preference systematically differ by culture and age stage."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

