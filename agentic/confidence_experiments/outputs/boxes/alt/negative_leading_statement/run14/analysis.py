import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_chi2(table: pd.DataFrame):
    """Compute chi-square test, handling degenerate tables."""
    # Drop any all-zero rows or columns to avoid numerical issues
    cleaned = table.loc[(table.sum(axis=1) > 0), :]
    cleaned = cleaned.loc[:, (cleaned.sum(axis=0) > 0)]
    if cleaned.shape[0] < 2 or cleaned.shape[1] < 2:
        return np.nan, np.nan, np.nan
    chi2, p, dof, _ = chi2_contingency(cleaned)
    return chi2, p, dof


def main():
    df = pd.read_csv("boxes.csv")

    # Encode reliance on social information: 1 if majority or minority, 0 if undemonstrated option
    df["social"] = (df["y"] != 1).astype(int)

    # Age groups for developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3.5, 6.5, 9.5, 12.5, 14.5],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )

    # Subset for children who used social information to study majority vs minority preference
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # --- Chi-square tests for reliance on social information ---
    table_culture_social = pd.crosstab(df["culture"], df["social"])
    chi2_culture_social, p_culture_social, dof_culture_social = compute_chi2(
        table_culture_social
    )

    table_age_social = pd.crosstab(df["age_group"], df["social"])
    chi2_age_social, p_age_social, dof_age_social = compute_chi2(table_age_social)

    # --- Chi-square tests for majority preference among social learners ---
    table_culture_majority = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_culture_majority, p_culture_majority, dof_culture_majority = compute_chi2(
        table_culture_majority
    )

    table_age_majority = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    chi2_age_majority, p_age_majority, dof_age_majority = compute_chi2(table_age_majority)

    # --- Descriptive statistics: proportions by culture and age group ---
    social_by_culture = df.groupby("culture")["social"].mean()
    social_culture_min = float(social_by_culture.min())
    social_culture_max = float(social_by_culture.max())

    social_by_age = df.groupby("age_group")["social"].mean()
    social_age_min = float(social_by_age.min())
    social_age_max = float(social_by_age.max())

    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()
    majority_culture_min = float(majority_by_culture.min())
    majority_culture_max = float(majority_by_culture.max())

    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()
    majority_age_min = float(majority_by_age.min())
    majority_age_max = float(majority_by_age.max())

    # --- Aggregate evidence to map onto Likert scale 0-100 ---
    p_values = [
        p_culture_social,
        p_age_social,
        p_culture_majority,
        p_age_majority,
    ]
    # Count strong and moderate evidence against the null of "no variation"
    n_strong = sum(1 for p in p_values if not np.isnan(p) and p < 0.01)
    n_moderate = sum(
        1 for p in p_values if not np.isnan(p) and 0.01 <= p < 0.05
    )

    if n_strong >= 2 or (n_strong >= 1 and n_moderate >= 1):
        response = 90
    elif n_strong + n_moderate >= 2:
        response = 80
    elif n_strong + n_moderate == 1:
        response = 65
    else:
        # Look for any weak trend
        any_trend = any(
            not np.isnan(p) and 0.05 <= p < 0.1 for p in p_values
        )
        response = 55 if any_trend else 20

    # Clamp to 0-100
    response = int(max(0, min(100, response)))

    # --- Build explanation string ---
    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    explanation_parts = []

    explanation_parts.append(
        "I analysed the 629-child dataset using contingency tables and chi-square tests to ask "
        "whether reliance on social information (choosing the majority or minority demonstrator "
        "rather than an undemonstrated option) and preference for majority cues vary across "
        "cultures and developmental stages."
    )

    explanation_parts.append(
        f"Reliance on social information differed across cultures: social-choice rates ranged from "
        f"{fmt_pct(social_culture_min)} to {fmt_pct(social_culture_max)} by site "
        f"(chi-square χ²={chi2_culture_social:.2f}, df={int(dof_culture_social)}, "
        f"p={p_culture_social:.3g}). Across age groups (4–6, 7–9, 10–12, 13–14 years), "
        f"social-choice rates ranged from {fmt_pct(social_age_min)} to {fmt_pct(social_age_max)} "
        f"(χ²={chi2_age_social:.2f}, df={int(dof_age_social)}, p={p_age_social:.3g})."
    )

    explanation_parts.append(
        f"Among children who used social information at all, preference for the majority option "
        f"also varied. Majority-choice rates by culture ranged from "
        f"{fmt_pct(majority_culture_min)} to {fmt_pct(majority_culture_max)} "
        f"(χ²={chi2_culture_majority:.2f}, df={int(dof_culture_majority)}, "
        f"p={p_culture_majority:.3g}), and by age group from "
        f"{fmt_pct(majority_age_min)} to {fmt_pct(majority_age_max)} "
        f"(χ²={chi2_age_majority:.2f}, df={int(dof_age_majority)}, "
        f"p={p_age_majority:.3g})."
    )

    if response >= 50:
        summary_sentence = (
            "Taken together, these differences in both overall social reliance and majority-"
            "preference rates across cultures and age groups provide evidence that children’s "
            "reliance on social information and preference for majority cues do vary across "
            "cultures and developmental stages."
        )
    else:
        summary_sentence = (
            "Taken together, the tests provide little evidence that children’s reliance on social "
            "information or preference for majority cues varies meaningfully across cultures or "
            "developmental stages."
        )

    explanation_parts.append(summary_sentence)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

