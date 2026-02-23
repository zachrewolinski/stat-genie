import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logistic_regression(
    data_frame: pd.DataFrame, predictor_columns: list[str], response_column: str
):
    design_matrix = sm.add_constant(
        data_frame[predictor_columns], has_constant="add"
    )
    response = data_frame[response_column]
    model = sm.Logit(response, design_matrix)
    result = model.fit(disp=False)
    return result


def main() -> None:
    data_path = Path("crofoot.csv")
    data_frame = pd.read_csv(data_path)

    # Construct key predictors reflecting the research question.
    # Relative group size: positive when the focal group is larger.
    data_frame["size_diff"] = data_frame["n_focal"] - data_frame["n_other"]
    # Relative location: negative when the focal group is closer to its home-range centre.
    data_frame["loc_diff"] = data_frame["dist_focal"] - data_frame["dist_other"]

    # Fit logistic regression models for the binary outcome "win".
    full_result = fit_logistic_regression(
        data_frame, ["size_diff", "loc_diff"], "win"
    )
    size_only_result = fit_logistic_regression(
        data_frame, ["size_diff"], "win"
    )
    location_only_result = fit_logistic_regression(
        data_frame, ["loc_diff"], "win"
    )

    full_params = full_result.params.to_dict()
    full_pvalues = full_result.pvalues.to_dict()
    full_odds_ratios = {name: float(np.exp(coef)) for name, coef in full_params.items()}

    size_only_params = size_only_result.params.to_dict()
    size_only_pvalues = size_only_result.pvalues.to_dict()
    size_only_odds_ratios = {
        name: float(np.exp(coef)) for name, coef in size_only_params.items()
    }

    location_only_params = location_only_result.params.to_dict()
    location_only_pvalues = location_only_result.pvalues.to_dict()
    location_only_odds_ratios = {
        name: float(np.exp(coef)) for name, coef in location_only_params.items()
    }

    alpha = 0.05

    size_pvalue_full = float(full_pvalues.get("size_diff", np.nan))
    location_pvalue_full = float(full_pvalues.get("loc_diff", np.nan))

    size_pvalue_only = float(size_only_pvalues.get("size_diff", np.nan))
    location_pvalue_only = float(location_only_pvalues.get("loc_diff", np.nan))

    size_significant = (size_pvalue_full < alpha) or (size_pvalue_only < alpha)
    location_significant = (location_pvalue_full < alpha) or (
        location_pvalue_only < alpha
    )

    # Map statistical evidence to a 0–100 scale.
    if size_significant or location_significant:
        base_score = 70
        if size_significant and location_significant:
            base_score = 85

        strongest_signal = min(
            [
                p
                for p in [
                    size_pvalue_full,
                    location_pvalue_full,
                    size_pvalue_only,
                    location_pvalue_only,
                ]
                if not np.isnan(p)
            ]
        )
        if strongest_signal < 0.001:
            base_score += 10
        elif strongest_signal < 0.01:
            base_score += 7
        elif strongest_signal < 0.05:
            base_score += 5

        response_score = max(0, min(100, int(round(base_score))))
        qualitative_answer = "Yes"
    else:
        response_score = 20
        qualitative_answer = "No"

    number_of_observations = int(data_frame.shape[0])

    explanation_parts: list[str] = []
    explanation_parts.append(
        f"I analysed {number_of_observations} intergroup contests using logistic regression, "
        "modelling the probability that the focal group won (win = 1) as a function of "
        "relative group size and contest location."
    )
    explanation_parts.append(
        "Relative group size was defined as n_focal − n_other, so positive values indicate "
        "that the focal group was larger. Contest location was captured as dist_focal − dist_other, "
        "so negative values mean the focal group was closer to its home-range centre than the opposing group."
    )

    explanation_parts.append(
        "In the model including both predictors, the coefficient for relative group size "
        f"was {full_params.get('size_diff', float('nan')):.3f} "
        f"(odds ratio = {full_odds_ratios.get('size_diff', float('nan')):.2f}, "
        f"p = {size_pvalue_full:.3g}), while the coefficient for contest location "
        f"was {full_params.get('loc_diff', float('nan')):.3f} "
        f"(odds ratio = {full_odds_ratios.get('loc_diff', float('nan')):.2f}, "
        f"p = {location_pvalue_full:.3g})."
    )

    explanation_parts.append(
        "Univariate logistic regressions broadly supported these patterns. For relative group size alone, "
        f"the odds ratio was {size_only_odds_ratios.get('size_diff', float('nan')):.2f} "
        f"with p = {size_pvalue_only:.3g}; for contest location alone, the odds ratio was "
        f"{location_only_odds_ratios.get('loc_diff', float('nan')):.2f} "
        f"with p = {location_pvalue_only:.3g}."
    )

    if size_significant:
        explanation_parts.append(
            "These results provide statistically significant evidence (at the 0.05 level) "
            "that relative group size affects the probability of winning intergroup contests."
        )
    else:
        explanation_parts.append(
            "For relative group size, p-values exceed 0.05, so there is no statistically "
            "significant evidence that group-size differences affect the probability of winning "
            "in this sample."
        )

    if location_significant:
        explanation_parts.append(
            "The analyses also provide statistically significant evidence that contest location—"
            "whether the encounter occurs closer to the focal versus the opposing group's home-range centre—"
            "influences which group wins."
        )
    else:
        explanation_parts.append(
            "For contest location, p-values exceed 0.05, so there is no statistically significant "
            "evidence that proximity to the home-range centre affects the probability of winning "
            "in this dataset."
        )

    if qualitative_answer == "Yes":
        summary_clause = (
            "there is statistically significant evidence in this dataset that relative group size "
            "and/or contest location influence the probability of winning."
        )
    else:
        summary_clause = (
            "there is little to no statistically significant evidence in this dataset that "
            "relative group size or contest location influence the probability of winning."
        )

    explanation_parts.append(
        f"Overall, I summarise the research question as a '{qualitative_answer}' answer: "
        f"{summary_clause} This corresponds to a confidence score of {response_score} on a 0–100 scale "
        "(0 = strong 'No', 100 = strong 'Yes')."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as file:
        json.dump(conclusion, file)


if __name__ == "__main__":
    main()
