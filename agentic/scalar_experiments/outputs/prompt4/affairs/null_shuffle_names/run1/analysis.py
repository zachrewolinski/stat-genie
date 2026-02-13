import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # According to info.json metadata:
    # - Column "age" encodes frequency of extramarital intercourse in the past year.
    # - Column "religiousness" is actually a yes/no indicator for whether there are children.
    affair_freq = df["age"]
    has_affair = (affair_freq > 0).astype(int)

    children_indicator = df["religiousness"].astype(str).str.lower()
    has_children = children_indicator == "yes"

    # Basic group summaries
    df_summary = (
        pd.DataFrame(
            {
                "has_affair": has_affair,
                "any_affair": has_affair,
                "affair_freq": affair_freq,
                "has_children": has_children,
            }
        )
    )

    group_stats = (
        df_summary.groupby("has_children")
        .agg(
            n=("any_affair", "size"),
            n_affairs=("any_affair", "sum"),
            prop_affair=("any_affair", "mean"),
            mean_freq=("affair_freq", "mean"),
        )
        .reset_index()
    )

    # 2x2 contingency table for chi-squared test on any affair vs children
    # rows: has_children False/True, cols: no affair / any affair
    contingency = np.zeros((2, 2), dtype=int)
    for i, has_children_flag in enumerate([False, True]):
        subset = df_summary[df_summary["has_children"] == has_children_flag]
        no_affair = (subset["any_affair"] == 0).sum()
        any_affair = (subset["any_affair"] == 1).sum()
        contingency[i, 0] = no_affair
        contingency[i, 1] = any_affair

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Decide an interpretable effect measure: difference in proportions
    prop_no_children = group_stats.loc[group_stats["has_children"] == False, "prop_affair"].iloc[0]  # noqa: E712
    prop_children = group_stats.loc[group_stats["has_children"] == True, "prop_affair"].iloc[0]  # noqa: E712
    diff_prop = prop_no_children - prop_children

    # Map evidence to a 0-100 response score, where higher means stronger "Yes:
    # having children decreases engagement in extramarital affairs".
    # Heuristic: combine direction (diff_prop) and significance (p-value).
    if p_value < 0.001 and diff_prop > 0:
        response_score = 95
    elif p_value < 0.01 and diff_prop > 0:
        response_score = 85
    elif p_value < 0.05 and diff_prop > 0:
        response_score = 75
    elif diff_prop > 0 and p_value < 0.1:
        response_score = 65
    elif diff_prop > 0:
        response_score = 55
    elif abs(diff_prop) < 0.01 or p_value > 0.5:
        # Essentially no detectable effect
        response_score = 50
    elif diff_prop < 0 and p_value < 0.05:
        # Evidence that having children is associated with *more* affairs
        response_score = 20
    else:
        response_score = 40

    # Build a concise explanation string summarizing the evidence.
    explanation_parts = []
    explanation_parts.append(
        "Using the survey data, I treated the 'age' column (per metadata, the coded "
        "frequency of extramarital intercourse in the past year) as an indicator of "
        "whether each respondent engaged in any extramarital affair (nonzero vs. zero)."
    )
    explanation_parts.append(
        "The 'religiousness' column is documented as a yes/no variable for whether "
        "there are children in the marriage, so I compared affair rates between couples "
        "with and without children."
    )

    row_no_children = group_stats[group_stats["has_children"] == False].iloc[0]  # noqa: E712
    row_children = group_stats[group_stats["has_children"] == True].iloc[0]  # noqa: E712

    explanation_parts.append(
        f"Among couples without children, {int(row_no_children['n_affairs'])} out of "
        f"{int(row_no_children['n'])} respondents ({row_no_children['prop_affair']:.1%}) "
        "reported at least one extramarital affair in the last year."
    )
    explanation_parts.append(
        f"Among couples with children, {int(row_children['n_affairs'])} out of "
        f"{int(row_children['n'])} respondents ({row_children['prop_affair']:.1%}) "
        "reported at least one affair."
    )
    explanation_parts.append(
        f"This corresponds to an absolute difference in affair prevalence of "
        f"{diff_prop:.1%} (no-children minus children). A chi-squared test on the "
        f"2×2 table of children-by-affair yields χ² = {chi2:.2f} with p = {p_value:.4g}."
    )

    if diff_prop > 0:
        direction_sentence = (
            "Because affair rates are higher among couples without children than among "
            "those with children, the data support the view that having children is "
            "associated with a modest decrease in engagement in extramarital affairs."
        )
    elif diff_prop < 0:
        direction_sentence = (
            "Because affair rates are actually higher among couples with children, the "
            "data do not support the claim that having children decreases affairs; if "
            "anything, the association is in the opposite direction."
        )
    else:
        direction_sentence = (
            "The affair rates are nearly identical between the two groups, so the data "
            "do not show that having children meaningfully changes engagement in "
            "extramarital affairs."
        )
    explanation_parts.append(direction_sentence)

    explanation_parts.append(
        "This conclusion is based on simple group comparisons and a chi-squared test; "
        "it does not adjust for potential confounders such as age, years married, or "
        "marital satisfaction, so the results should be interpreted as evidence of "
        "association rather than causal effects."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write to conclusion.txt as required.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

