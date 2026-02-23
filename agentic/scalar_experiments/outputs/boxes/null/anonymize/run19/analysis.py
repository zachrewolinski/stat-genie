import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def safe_pearsonr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Pearson correlation, handling constant-input edge cases."""
    x_non_null = x.dropna()
    y_non_null = y.dropna()
    aligned = pd.concat([x_non_null, y_non_null], axis=1).dropna()
    if aligned.shape[0] < 3:
        return 0.0, 1.0
    if aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return 0.0, 1.0
    r, p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(r), float(p)


def cramers_v_from_chi2(chi2: float, table: pd.DataFrame) -> float:
    """Compute Cramer's V effect size from chi-square and contingency table."""
    n = float(table.to_numpy().sum())
    if n == 0:
        return 0.0
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * denom)))


def chi2_and_cramers(table: pd.DataFrame) -> tuple[float, float, float]:
    """Chi-square test of independence with Cramer's V, handling edge cases."""
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0, 0.0
    chi2, p, _, _ = stats.chi2_contingency(table)
    v = cramers_v_from_chi2(chi2, table)
    return float(chi2), float(p), float(v)


def evidence_from_test(p: float, effect: float) -> int:
    """
    Map p-value and effect size to an evidence score in [0, 25].

    Higher scores correspond to stronger, more reliable relationships.
    """
    p = float(p)
    effect = abs(float(effect))
    if p < 0.001 and effect >= 0.30:
        return 25
    if p < 0.01 and effect >= 0.20:
        return 20
    if p < 0.05 and effect >= 0.10:
        return 15
    if p < 0.05:
        return 10
    if p < 0.10 and effect >= 0.10:
        return 5
    return 0


def significance_phrase(p: float) -> str:
    """Qualitative description of strength of evidence from a p-value."""
    p = float(p)
    if p < 0.001:
        return "strong evidence of a relationship"
    if p < 0.01:
        return "clear evidence of a relationship"
    if p < 0.05:
        return "some evidence of a relationship"
    if p < 0.10:
        return "weak, marginal evidence of a relationship"
    return "little evidence of a relationship"


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Variable meanings based on info.json:
    # feature1: 1 = undemonstrated option (no social information),
    #           2 = majority option,
    #           3 = minority option.
    # feature3: age in years.
    # feature5: site / culture identifier (1–8).

    # Social reliance: following either majority or minority demonstrators.
    df["social_reliance"] = (df["feature1"] != 1).astype(int)

    # Majority preference among those who use social information.
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df_social = df[df["social_reliance"] == 1].copy()

    n_total = int(df.shape[0])
    n_sites = int(df["feature5"].nunique())

    # --- Age effects ---
    r_age_social, p_age_social = safe_pearsonr(df["feature3"], df["social_reliance"])
    score_age_social = evidence_from_test(p_age_social, r_age_social)

    r_age_major, p_age_major = safe_pearsonr(df_social["feature3"], df_social["majority_choice"])
    score_age_major = evidence_from_test(p_age_major, r_age_major)

    # --- Cultural effects (site differences) ---
    ct_social = pd.crosstab(df["feature5"], df["social_reliance"])
    chi2_cult_social, p_cult_social, v_cult_social = chi2_and_cramers(ct_social)
    score_cult_social = evidence_from_test(p_cult_social, v_cult_social)

    ct_major = pd.crosstab(df_social["feature5"], df_social["majority_choice"])
    chi2_cult_major, p_cult_major, v_cult_major = chi2_and_cramers(ct_major)
    score_cult_major = evidence_from_test(p_cult_major, v_cult_major)

    # Combine evidence across four tests into a 0–100 Likert score.
    total_score = score_age_social + score_age_major + score_cult_social + score_cult_major
    response_score = int(max(0, min(100, round(total_score))))

    # Directional interpretations for age effects.
    if r_age_social > 0:
        age_social_direction = (
            "older children were more likely than younger children "
            "to rely on social information (i.e., to follow either majority or minority cues)"
        )
    elif r_age_social < 0:
        age_social_direction = (
            "older children were less likely than younger children "
            "to rely on social information"
        )
    else:
        age_social_direction = (
            "there was no discernible directional trend in social reliance across age"
        )

    if r_age_major > 0:
        age_major_direction = (
            "among children who used social information, older children were more likely "
            "to follow the majority demonstrators rather than the minority"
        )
    elif r_age_major < 0:
        age_major_direction = (
            "among children who used social information, older children were less likely "
            "to follow the majority demonstrators"
        )
    else:
        age_major_direction = (
            "among social learners, there was no discernible directional trend in "
            "majority preference across age"
        )

    # Qualitative descriptions of evidence.
    age_social_sig = significance_phrase(p_age_social)
    age_major_sig = significance_phrase(p_age_major)
    cult_social_sig = significance_phrase(p_cult_social)
    cult_major_sig = significance_phrase(p_cult_major)

    explanation_lines = []
    explanation_lines.append(
        f"This analysis used N = {n_total} children from {n_sites} cultural sites "
        "to ask whether children's reliance on social information and preference for "
        "majority cues vary across cultures and developmental stages (age)."
    )
    explanation_lines.append(
        "Outcome choices were coded as: 1 = undemonstrated option (no social information), "
        "2 = majority option, and 3 = minority option. I defined 'social reliance' as "
        "choosing either the majority or minority option (feature1 != 1), and 'majority "
        "preference' as choosing the majority option (feature1 == 2) among those who relied "
        "on social information."
    )

    # Age-related findings.
    explanation_lines.append(
        f"For social reliance, the correlation between age (years) and social reliance "
        f"was r = {r_age_social:.3f}, p = {p_age_social:.3g}, providing "
        f"{age_social_sig}. In substantive terms, {age_social_direction}."
    )
    explanation_lines.append(
        f"For majority preference among social learners, the correlation between age and "
        f"majority choice was r = {r_age_major:.3f}, p = {p_age_major:.3g}, providing "
        f"{age_major_sig}. In substantive terms, {age_major_direction}."
    )

    # Cultural findings.
    explanation_lines.append(
        f"To assess cultural differences, I conducted chi-square tests of independence "
        f"across sites. For social reliance, the site × social-reliance contingency table "
        f"yielded χ² = {chi2_cult_social:.3f}, p = {p_cult_social:.3g}, "
        f"Cramer's V = {v_cult_social:.3f}, indicating {cult_social_sig} that "
        "children's tendency to use social information varies across cultural sites."
    )
    explanation_lines.append(
        f"For majority preference among social learners, the site × majority-choice "
        f"contingency table yielded χ² = {chi2_cult_major:.3f}, p = {p_cult_major:.3g}, "
        f"Cramer's V = {v_cult_major:.3f}, indicating {cult_major_sig} that the "
        "strength of children's majority preference differs across cultures."
    )

    # Overall conclusion mapped to Likert scale.
    if response_score >= 60:
        overall_statement = (
            "Taken together, these results indicate that children's reliance on social "
            "information and their preference for majority cues do vary meaningfully "
            "across both developmental stages and cultural contexts."
        )
    elif response_score >= 40:
        overall_statement = (
            "Taken together, the results suggest some variability in children's reliance "
            "on social information and majority preference across age and cultures, but "
            "the evidence is mixed or only modest in strength."
        )
    else:
        overall_statement = (
            "Overall, the analyses provide limited evidence that children's reliance on "
            "social information or their majority preference systematically vary across "
            "developmental stages or cultural sites in this dataset."
        )

    explanation_lines.append(overall_statement)
    explanation_lines.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"these combined findings correspond to a response of {response_score}."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

