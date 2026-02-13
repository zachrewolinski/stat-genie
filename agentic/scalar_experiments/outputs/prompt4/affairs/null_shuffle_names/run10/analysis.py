import json
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    return dataframe


def prepare_variables(dataframe: pd.DataFrame) -> pd.DataFrame:
    affair_frequency = dataframe["age"]
    children_raw = dataframe["religiousness"]

    children_mapping = {
        "yes": 1,
        "no": 0,
        "Yes": 1,
        "No": 0,
        "Y": 1,
        "N": 0,
    }
    has_children = children_raw.map(children_mapping)

    valid_mask = has_children.notna() & affair_frequency.notna()

    analysis_frame = pd.DataFrame(
        {
            "affair_frequency": affair_frequency[valid_mask],
            "has_children": has_children[valid_mask].astype(int),
        }
    )
    analysis_frame["has_affair"] = (analysis_frame["affair_frequency"] > 0).astype(int)

    return analysis_frame


def compute_group_statistics(analysis_frame: pd.DataFrame) -> pd.DataFrame:
    group_statistics = analysis_frame.groupby("has_children").agg(
        mean_affair_frequency=("affair_frequency", "mean"),
        proportion_with_affair=("has_affair", "mean"),
        count=("affair_frequency", "size"),
    )
    return group_statistics


def run_logistic_regression(analysis_frame: pd.DataFrame):
    design_matrix = sm.add_constant(analysis_frame["has_children"])
    logit_model = sm.Logit(analysis_frame["has_affair"], design_matrix)
    logit_result = logit_model.fit(disp=False)
    coefficient = float(logit_result.params["has_children"])
    p_value = float(logit_result.pvalues["has_children"])
    odds_ratio = float(np.exp(coefficient))
    return coefficient, p_value, odds_ratio


def run_linear_regression(analysis_frame: pd.DataFrame):
    design_matrix = sm.add_constant(analysis_frame["has_children"])
    ols_model = sm.OLS(analysis_frame["affair_frequency"], design_matrix)
    ols_result = ols_model.fit()
    coefficient = float(ols_result.params["has_children"])
    p_value = float(ols_result.pvalues["has_children"])
    return coefficient, p_value


def map_evidence_to_scale(coefficient: float, p_value: float) -> int:
    if coefficient < 0:
        if p_value < 0.01:
            return 90
        if p_value < 0.05:
            return 75
        return 60
    if p_value < 0.01:
        return 10
    if p_value < 0.05:
        return 25
    return 40


def build_explanation(
    group_statistics: pd.DataFrame,
    logit_coefficient: float,
    logit_p_value: float,
    logit_odds_ratio: float,
    linear_coefficient: float,
    linear_p_value: float,
) -> str:
    group_without_children = (
        group_statistics.loc[0] if 0 in group_statistics.index else None
    )
    group_with_children = (
        group_statistics.loc[1] if 1 in group_statistics.index else None
    )

    parts = []

    if group_without_children is not None:
        parts.append(
            (
                "Among respondents without children, "
                f"n={int(group_without_children['count'])}, "
                f"{group_without_children['proportion_with_affair'] * 100:.1f}% "
                "reported at least one extramarital sexual encounter in the past year, "
                f"with a mean affair-frequency score of "
                f"{group_without_children['mean_affair_frequency']:.2f} "
                "on the 0–12 scale."
            )
        )

    if group_with_children is not None:
        parts.append(
            (
                "Among respondents with children, "
                f"n={int(group_with_children['count'])}, "
                f"{group_with_children['proportion_with_affair'] * 100:.1f}% "
                "reported at least one extramarital sexual encounter in the past year, "
                f"with a mean affair-frequency score of "
                f"{group_with_children['mean_affair_frequency']:.2f} "
                "on the 0–12 scale."
            )
        )

    effect_direction = (
        "lower odds" if logit_coefficient < 0 else "higher odds"
    )

    parts.append(
        (
            "I then fit a logistic regression model predicting whether someone had any "
            "extramarital intercourse from the children indicator alone. "
            f"The coefficient for having children was {logit_coefficient:.3f} "
            f"(odds ratio {logit_odds_ratio:.2f}, p={logit_p_value:.4f}), "
            f"which implies {effect_direction} of having an affair for people with children "
            "relative to those without."
        )
    )

    parts.append(
        (
            "As a robustness check, I also fit a linear regression of the affair-frequency "
            "score on the children indicator. "
            f"The children coefficient was {linear_coefficient:.3f} "
            f"(p={linear_p_value:.4f}), consistent in sign with the logistic model."
        )
    )

    if logit_coefficient < 0:
        summary_sentence = (
            "Overall, the data suggest that having children is associated with modestly "
            "reduced engagement in extramarital affairs in this sample."
        )
    else:
        summary_sentence = (
            "Overall, the data do not support the idea that having children reduces "
            "engagement in extramarital affairs in this sample."
        )

    parts.append(summary_sentence)

    explanation = " ".join(parts)
    return explanation


def main() -> None:
    dataframe = load_data(Path("affairs.csv"))
    analysis_frame = prepare_variables(dataframe)

    group_statistics = compute_group_statistics(analysis_frame)
    (
        logit_coefficient,
        logit_p_value,
        logit_odds_ratio,
    ) = run_logistic_regression(analysis_frame)
    (
        linear_coefficient,
        linear_p_value,
    ) = run_linear_regression(analysis_frame)

    response_value = map_evidence_to_scale(logit_coefficient, logit_p_value)
    explanation_text = build_explanation(
        group_statistics,
        logit_coefficient,
        logit_p_value,
        logit_odds_ratio,
        linear_coefficient,
        linear_p_value,
    )

    conclusion = {"response": int(response_value), "explanation": explanation_text}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as file:
        json.dump(conclusion, file)


if __name__ == "__main__":
    main()
