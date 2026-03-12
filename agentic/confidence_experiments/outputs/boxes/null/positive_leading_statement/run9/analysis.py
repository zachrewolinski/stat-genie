import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def chi_square_summary(table: pd.DataFrame):
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, expected


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Define derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    # Define developmental stage bins (age groups)
    age_bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
    )
    social_df["age_group"] = pd.cut(
        social_df["age"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
    )

    # Overall rates
    overall_social_rate = df["social"].mean()
    overall_majority_rate = social_df["majority_choice"].mean()

    # Group-level rates by age and culture
    social_by_age = df.groupby("age_group")["social"].mean()
    social_by_culture = df.groupby("culture")["social"].mean()
    majority_by_age = social_df.groupby("age_group")["majority_choice"].mean()
    majority_by_culture = social_df.groupby("culture")["majority_choice"].mean()

    # Chi-square tests
    social_age_table = pd.crosstab(df["age_group"], df["social"])
    social_culture_table = pd.crosstab(df["culture"], df["social"])
    majority_age_table = pd.crosstab(social_df["age_group"], social_df["majority_choice"])
    majority_culture_table = pd.crosstab(social_df["culture"], social_df["majority_choice"])

    chi2_social_age, p_social_age, dof_social_age, _ = chi_square_summary(social_age_table)
    chi2_social_culture, p_social_culture, dof_social_culture, _ = chi_square_summary(
        social_culture_table
    )
    chi2_majority_age, p_majority_age, dof_majority_age, _ = chi_square_summary(
        majority_age_table
    )
    chi2_majority_culture, p_majority_culture, dof_majority_culture, _ = chi_square_summary(
        majority_culture_table
    )

    # Determine statistical significance flags
    sig_social_age = p_social_age < 0.05
    sig_social_culture = p_social_culture < 0.05
    sig_majority_age = p_majority_age < 0.05
    sig_majority_culture = p_majority_culture < 0.05

    num_sig = sum(
        [
            sig_social_age,
            sig_social_culture,
            sig_majority_age,
            sig_majority_culture,
        ]
    )

    # Effect size proxies: range of probabilities across groups
    def safe_range(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.max() - series.min())

    effect_social_age = safe_range(social_by_age)
    effect_social_culture = safe_range(social_by_culture)
    effect_majority_age = safe_range(majority_by_age)
    effect_majority_culture = safe_range(majority_by_culture)

    max_effect_range = max(
        effect_social_age,
        effect_social_culture,
        effect_majority_age,
        effect_majority_culture,
    )

    # Map evidence strength to a 0–100 Likert-style scalar
    if num_sig == 0:
        # No statistically significant evidence that outcomes vary by age or culture.
        # Use effect size to distinguish between "strong no" and "weak/uncertain no".
        if max_effect_range < 0.05:
            response_value = 10
        elif max_effect_range < 0.10:
            response_value = 25
        else:
            response_value = 40
        answer_label = "No"
    else:
        # At least some statistically significant evidence for variation.
        if num_sig >= 3 and max_effect_range >= 0.20:
            base = 85
        elif num_sig >= 2 and max_effect_range >= 0.10:
            base = 75
        else:
            base = 65
        # Add a small adjustment based on effect size, capped to keep within [0, 100].
        response_value = base + int(min(15, max_effect_range * 50))
        response_value = max(0, min(100, response_value))
        answer_label = "Yes"

    # Build textual explanation
    lines = []
    lines.append(
        "Research question: Do children’s reliance on social information and preference for "
        "majority cues vary across cultures and developmental stages (age)?"
    )
    lines.append(
        "Operationalisation: I treat reliance on social information as choosing any demonstrated "
        "option (majority or minority; y = 2 or 3) versus the undemonstrated option (y = 1). "
        "Preference for majority cues is defined, among socially informed choices (y in {2, 3}), "
        "as choosing the majority option (y = 2) rather than the minority option (y = 3)."
    )
    lines.append(
        f"Overall, {overall_social_rate*100:.1f}% of children chose a socially informed option, "
        f"and among these social choices, {overall_majority_rate*100:.1f}% followed the majority "
        "demonstrators rather than the minority."
    )

    if not social_by_age.empty:
        lines.append(
            "By developmental stage (age groups 4–6, 7–9, 10–12, 13–14), reliance on social "
            f"information ranged from {social_by_age.min()*100:.1f}% to "
            f"{social_by_age.max()*100:.1f}% across age groups, while majority-following among "
            f"social choices ranged from {majority_by_age.min()*100:.1f}% to "
            f"{majority_by_age.max()*100:.1f}%."
        )

    if not social_by_culture.empty:
        lines.append(
            "Across the eight cultural sites, reliance on social information ranged from "
            f"{social_by_culture.min()*100:.1f}% to {social_by_culture.max()*100:.1f}%, and "
            "majority-following among social choices ranged from "
            f"{majority_by_culture.min()*100:.1f}% to {majority_by_culture.max()*100:.1f}%."
        )

    # Add statistical test summaries
    def test_line(name: str, chi2_val: float, dof: int, p_val: float) -> str:
        sig_text = "statistically significant" if p_val < 0.05 else "not statistically significant"
        return (
            f"{name}: chi-square({dof}) = {chi2_val:.2f}, p {format_p(p_val)} "
            f"({sig_text} at α = 0.05)."
        )

    lines.append(
        test_line(
            "Social vs asocial choices by age group",
            chi2_social_age,
            dof_social_age,
            p_social_age,
        )
    )
    lines.append(
        test_line(
            "Social vs asocial choices by culture",
            chi2_social_culture,
            dof_social_culture,
            p_social_culture,
        )
    )
    lines.append(
        test_line(
            "Majority vs minority social choices by age group",
            chi2_majority_age,
            dof_majority_age,
            p_majority_age,
        )
    )
    lines.append(
        test_line(
            "Majority vs minority social choices by culture",
            chi2_majority_culture,
            dof_majority_culture,
            p_majority_culture,
        )
    )

    if answer_label == "Yes":
        lines.append(
            "Taken together, these results provide statistically significant evidence that both "
            "children’s reliance on social information and their preference for majority cues "
            "vary across cultures and developmental stages. At least one of the chi-square tests "
            "for age or culture is significant at the 0.05 level, and the observed differences in "
            f"proportions across groups (maximum range ≈ {max_effect_range*100:.1f} percentage "
            "points) are substantively meaningful."
        )
    else:
        lines.append(
            "Overall, the chi-square tests do not provide strong statistical evidence that "
            "children’s reliance on social information or their preference for majority cues "
            "systematically vary across age groups or cultures. Any observed differences in "
            f"group-level proportions (maximum range ≈ {max_effect_range*100:.1f} percentage "
            "points) are modest relative to sampling variability, so I interpret the evidence as "
            "insufficient to support robust age- or culture-related variation in these social "
            "learning tendencies."
        )

    lines.append(
        f"Conclusion on the research question: Based on these analyses, my answer is '{answer_label}'. "
        f"The corresponding scalar response on the 0–100 scale is {response_value}, where higher "
        "values indicate stronger support for the existence of age- and culture-related variation "
        "in children’s reliance on social information and majority preference."
    )

    explanation = "\n".join(lines)

    result = {"response": int(response_value), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

