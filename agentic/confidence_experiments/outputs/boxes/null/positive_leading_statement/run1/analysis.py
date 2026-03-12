import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ChiSquareResult:
    chi2: float
    p_value: float
    dof: int
    cramers_v: float


def chi_square_with_cramers_v(table: pd.DataFrame) -> ChiSquareResult:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.values.sum()
    r, c = table.shape
    if n == 0 or r <= 1 or c <= 1:
        v = 0.0
    else:
        v = float(np.sqrt(chi2 / (n * (min(r - 1, c - 1)))))
    return ChiSquareResult(chi2=float(chi2), p_value=float(p), dof=int(dof), cramers_v=v)


def evidence_score_from_result(result: ChiSquareResult) -> float:
    """Map chi-square result to an evidence score in [0, 1]."""
    p = result.p_value
    v = result.cramers_v

    if p < 0.01:
        sig_score = 1.0
    elif p < 0.05:
        sig_score = 0.7
    elif p < 0.1:
        sig_score = 0.3
    else:
        sig_score = 0.0

    # Scale effect size assuming Cramer's V in [0, 0.5+] where 0.5 is very strong
    effect_score = min(v / 0.5, 1.0)

    # Combine significance and effect size
    return 0.6 * sig_score + 0.4 * effect_score


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define key derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Age groups representing developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[4, 7, 10, 13, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        right=False,
        include_lowest=True,
    )

    # Overall descriptive statistics
    n = len(df)
    social_rate = df["social"].mean()
    majority_rate_among_social = df.loc[df["social"] == 1, "majority_choice"].mean()

    # Descriptive by age group and culture
    social_by_age = df.groupby("age_group")["social"].mean()
    social_by_culture = df.groupby("culture")["social"].mean()

    df_social = df[df["social"] == 1].copy()
    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()

    # Inferential tests: chi-square with Cramer's V
    social_age_table = pd.crosstab(df["social"], df["age_group"])
    social_culture_table = pd.crosstab(df["social"], df["culture"])
    majority_age_table = pd.crosstab(df_social["majority_choice"], df_social["age_group"])
    majority_culture_table = pd.crosstab(df_social["majority_choice"], df_social["culture"])

    social_age_res = chi_square_with_cramers_v(social_age_table)
    social_culture_res = chi_square_with_cramers_v(social_culture_table)
    majority_age_res = chi_square_with_cramers_v(majority_age_table)
    majority_culture_res = chi_square_with_cramers_v(majority_culture_table)

    # Aggregate evidence that reliance on social info and majority preference
    # vary across cultures and developmental stages.
    evidence_components = [
        evidence_score_from_result(social_age_res),
        evidence_score_from_result(social_culture_res),
        evidence_score_from_result(majority_age_res),
        evidence_score_from_result(majority_culture_res),
    ]

    overall_evidence_score = float(np.mean(evidence_components))
    response_score = int(round(100 * overall_evidence_score))

    # Build explanation text
    explanation_parts = []
    explanation_parts.append(
        f"The dataset contains {n} children aged 4–14 across 8 cultural sites. "
        f"Overall, children relied on social information (choosing a demonstrated option) in "
        f"{social_rate * 100:.1f}% of trials, and when they followed social information they chose "
        f"the majority option in {majority_rate_among_social * 100:.1f}% of cases."
    )

    explanation_parts.append(
        "Reliance on social information varies across developmental stages: the proportion of "
        "social choices by age group ranges from "
        f"{social_by_age.min() * 100:.1f}% to {social_by_age.max() * 100:.1f}% "
        f"(χ²={social_age_res.chi2:.2f}, df={social_age_res.dof}, p={social_age_res.p_value:.3f}, "
        f"Cramer's V={social_age_res.cramers_v:.3f}). "
        "Across cultural sites, the proportion of social choices ranges from "
        f"{social_by_culture.min() * 100:.1f}% to {social_by_culture.max() * 100:.1f}% "
        f"(χ²={social_culture_res.chi2:.2f}, df={social_culture_res.dof}, "
        f"p={social_culture_res.p_value:.3f}, Cramer's V={social_culture_res.cramers_v:.3f})."
    )

    explanation_parts.append(
        "Among children who used social information at all, preference for the majority option also "
        "varies with age and culture. Majority choices by age group range from "
        f"{majority_by_age.min() * 100:.1f}% to {majority_by_age.max() * 100:.1f}% "
        f"(χ²={majority_age_res.chi2:.2f}, df={majority_age_res.dof}, "
        f"p={majority_age_res.p_value:.3f}, Cramer's V={majority_age_res.cramers_v:.3f}), and across "
        f"cultures from {majority_by_culture.min() * 100:.1f}% to {majority_by_culture.max() * 100:.1f}% "
        f"(χ²={majority_culture_res.chi2:.2f}, df={majority_culture_res.dof}, "
        f"p={majority_culture_res.p_value:.3f}, Cramer's V={majority_culture_res.cramers_v:.3f})."
    )

    if response_score >= 60:
        summary_statement = (
            "Taken together, these patterns provide overall statistical evidence that both children's "
            "reliance on social information and their preference for majority cues vary across cultures "
            "and developmental stages."
        )
    elif response_score >= 40:
        summary_statement = (
            "Taken together, these patterns provide mixed but suggestive statistical evidence that "
            "children's reliance on social information and their preference for majority cues vary "
            "across cultures and developmental stages."
        )
    else:
        summary_statement = (
            "Taken together, these patterns provide limited statistical evidence that children's reliance "
            "on social information and their preference for majority cues vary across cultures and "
            "developmental stages."
        )

    explanation_parts.append(summary_statement)

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

