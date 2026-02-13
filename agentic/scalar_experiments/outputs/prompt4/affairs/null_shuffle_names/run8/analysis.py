import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_metadata(base_dir: Path) -> dict:
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def find_column_by_description(fields, substring: str) -> str:
    substring_lower = substring.lower()
    for field in fields:
        description = field.get("properties", {}).get("description", "") or ""
        if substring_lower in description.lower():
            return field["column"]
    raise ValueError(f"Could not find column with description containing: {substring!r}")


def compute_effect_strength(coef: float, p_value: float) -> int:
    """
    Map effect direction and significance to a 0–100 Likert-style score.

    0  = strong 'No, children do not decrease affairs'
    100 = strong 'Yes, children decrease affairs'
    """
    if np.isnan(coef) or np.isnan(p_value):
        return 50

    if coef < 0:
        # Having children is associated with fewer affairs
        if p_value < 0.01:
            return 90
        if p_value < 0.05:
            return 80
        if p_value < 0.1:
            return 65
        return 55

    if coef > 0:
        # Having children is associated with more affairs
        if p_value < 0.01:
            return 10
        if p_value < 0.05:
            return 20
        if p_value < 0.1:
            return 35
        return 45

    return 50


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    metadata = load_metadata(base_dir)
    fields = metadata["data_desc"]["fields"]

    # Use semantic descriptions from metadata rather than shuffled column names
    affair_column = find_column_by_description(
        fields, "How often engaged in extramarital sexual intercourse"
    )
    children_column = find_column_by_description(
        fields, "Are there children in the marriage"
    )

    data_path = base_dir / "affairs.csv"
    data = pd.read_csv(data_path)

    # Recode children indicator: yes/no -> 1/0
    children_raw = data[children_column].astype(str).str.lower()
    has_children = children_raw.map({"yes": 1, "no": 0})
    valid_mask = has_children.isin([0, 1])
    analysis_data = data.loc[valid_mask].copy()
    analysis_data["has_children"] = has_children[valid_mask]

    # Affair frequency variable as provided in metadata (0 = none, higher = more)
    analysis_data["affair_frequency"] = analysis_data[affair_column].astype(float)
    analysis_data["any_affair"] = (analysis_data["affair_frequency"] > 0).astype(int)

    group_stats = (
        analysis_data.groupby("has_children")["affair_frequency"]
        .agg(["mean", "median", "std", "count"])
        .to_dict(orient="index")
    )

    proportion_any = (
        analysis_data.groupby("has_children")["any_affair"].mean().to_dict()
    )

    # Two-sample test for difference in mean affair frequency (non-parametric)
    no_children_affairs = analysis_data.loc[
        analysis_data["has_children"] == 0, "affair_frequency"
    ]
    with_children_affairs = analysis_data.loc[
        analysis_data["has_children"] == 1, "affair_frequency"
    ]

    try:
        rank_stat, rank_p_value = stats.mannwhitneyu(
            no_children_affairs,
            with_children_affairs,
            alternative="two-sided",
        )
    except Exception:
        rank_stat, rank_p_value = np.nan, np.nan

    # Logistic regression for any affair ~ has_children
    logistic_coef = np.nan
    logistic_p_value = np.nan
    odds_ratio = np.nan

    try:
        predictors = sm.add_constant(analysis_data[["has_children"]])
        response = analysis_data["any_affair"]
        model = sm.Logit(response, predictors, missing="drop")
        result = model.fit(disp=False)
        logistic_coef = float(result.params["has_children"])
        logistic_p_value = float(result.pvalues["has_children"])
        odds_ratio = float(np.exp(logistic_coef))
    except Exception:
        logistic_coef = np.nan
        logistic_p_value = np.nan
        odds_ratio = np.nan

    response_score = compute_effect_strength(logistic_coef, logistic_p_value)

    # Build textual explanation
    mean_no_children = group_stats.get(0, {}).get("mean", float("nan"))
    mean_with_children = group_stats.get(1, {}).get("mean", float("nan"))
    prop_no_children = proportion_any.get(0, float("nan"))
    prop_with_children = proportion_any.get(1, float("nan"))

    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        "Using the provided metadata, I treated the variable described as "
        "'How often engaged in extramarital sexual intercourse during the past year' "
        "as the affair frequency outcome and the variable described as "
        "'Are there children in the marriage?' as the children indicator, "
        "since the column names in the CSV are shuffled."
    )
    explanation_parts.append(
        f"In total, {int(group_stats.get(0, {}).get('count', 0) + group_stats.get(1, {}).get('count', 0))} "
        "married individuals with valid data were analyzed."
    )
    explanation_parts.append(
        "I created two outcomes: a numeric affair frequency score (higher values = more frequent affairs) "
        "and a binary indicator of having any affair in the past year (score > 0). "
        "I then compared people with and without children."
    )
    explanation_parts.append(
        f"Mean affair frequency (0 = none) was approximately {mean_no_children:.3f} "
        "for those without children and "
        f"{mean_with_children:.3f} for those with children."
    )
    explanation_parts.append(
        f"The proportion who reported any extramarital affair was about {prop_no_children:.3f} "
        "without children versus "
        f"{prop_with_children:.3f} with children."
    )

    if not np.isnan(rank_p_value):
        explanation_parts.append(
            "A Mann–Whitney U test comparing the distributions of affair frequency between those with "
            f"and without children yielded a two-sided p-value of approximately {rank_p_value:.3f}."
        )

    if not np.isnan(logistic_coef):
        direction = (
            "lower" if logistic_coef < 0 else "higher" if logistic_coef > 0 else "similar"
        )
        explanation_parts.append(
            "I also fit a simple logistic regression with 'any affair' as the outcome and "
            "'having children' as the sole predictor. "
            f"The coefficient on having children was {logistic_coef:.3f}, corresponding to an odds ratio "
            f"of about {odds_ratio:.3f}, with a p-value of approximately {logistic_p_value:.3f}. "
            f"This indicates {direction} odds of having an affair among those with children compared to those without."
        )

    if response_score > 55:
        qualitative = (
            "the data provide some evidence that having children is associated with fewer extramarital affairs, "
            "though effect size and statistical significance should be considered together."
        )
    elif response_score < 45:
        qualitative = (
            "the data suggest that having children does not decrease extramarital affairs and may even be associated "
            "with equal or greater engagement in affairs."
        )
    else:
        qualitative = (
            "the data do not provide clear evidence that having children meaningfully decreases extramarital affairs."
        )

    explanation_parts.append(
        f"Mapping these results to a 0–100 scale, where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"I assign a score of {response_score}. In words, {qualitative}"
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    output_path = base_dir / "conclusion.txt"
    with output_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

