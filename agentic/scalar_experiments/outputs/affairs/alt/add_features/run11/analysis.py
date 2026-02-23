import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.copy()
    dataframe = dataframe.dropna(subset=["affairs", "children"])

    dataframe["any_affair"] = (dataframe["affairs"] > 0).astype(int)
    dataframe["children_yes"] = (
        dataframe["children"].astype(str).str.lower().eq("yes")
    ).astype(int)

    return dataframe


def compute_group_stats(dataframe: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean_affairs = dataframe.groupby("children")["affairs"].mean().to_dict()
    prop_any = dataframe.groupby("children")["any_affair"].mean().to_dict()
    return mean_affairs, prop_any


def fit_models(dataframe: pd.DataFrame):
    effects = {}

    # Unadjusted logistic regression on any affair
    logit_unadj = smf.logit("any_affair ~ children_yes", data=dataframe).fit(disp=False)
    coef_unadj = logit_unadj.params["children_yes"]
    p_unadj = logit_unadj.pvalues["children_yes"]
    or_unadj = float(np.exp(coef_unadj))
    effects["logit_unadjusted"] = (coef_unadj, p_unadj, or_unadj)

    # Adjusted logistic regression with standard covariates
    covariates = [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    model_data_adj = dataframe.dropna(
        subset=covariates + ["any_affair", "children_yes", "gender"]
    )
    logit_formula_adj = (
        "any_affair ~ children_yes + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_adj = smf.logit(logit_formula_adj, data=model_data_adj).fit(disp=False)
    coef_adj = logit_adj.params["children_yes"]
    p_adj = logit_adj.pvalues["children_yes"]
    or_adj = float(np.exp(coef_adj))
    effects["logit_adjusted"] = (coef_adj, p_adj, or_adj)

    # Poisson regression on affair counts
    poisson_formula = (
        "affairs ~ children_yes + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model_data_pois = dataframe.dropna(
        subset=covariates + ["affairs", "children_yes", "gender"]
    ).assign(affairs=lambda df_: df_["affairs"].astype(float))
    poisson_model = smf.glm(
        poisson_formula,
        data=model_data_pois,
        family=sm.families.Poisson(),
    ).fit()
    coef_pois = poisson_model.params["children_yes"]
    p_pois = poisson_model.pvalues["children_yes"]
    rr_pois = float(np.exp(coef_pois))
    effects["poisson"] = (coef_pois, p_pois, rr_pois)

    return effects


def map_evidence_to_score(
    effects: Dict[str, Tuple[float, float, float]],
    mean_affairs: Dict[str, float],
    prop_any: Dict[str, float],
) -> int:
    negative_sig = 0
    negative_nonsig = 0
    positive_sig = 0
    positive_nonsig = 0

    for coef, p_value, _ in effects.values():
        if coef < 0:
            if p_value < 0.05:
                negative_sig += 1
            else:
                negative_nonsig += 1
        elif coef > 0:
            if p_value < 0.05:
                positive_sig += 1
            else:
                positive_nonsig += 1

    if negative_sig >= 3:
        base_score = 90
    elif negative_sig == 2 and positive_sig == 0:
        base_score = 80
    elif negative_sig == 1 and positive_sig == 0:
        base_score = 65
    elif positive_sig >= 3:
        base_score = 10
    elif positive_sig == 2 and negative_sig == 0:
        base_score = 20
    elif positive_sig == 1 and negative_sig == 0:
        base_score = 35
    else:
        total_negative = negative_sig + negative_nonsig
        total_positive = positive_sig + positive_nonsig
        if total_negative > total_positive:
            base_score = 55
        elif total_positive > total_negative:
            base_score = 45
        else:
            base_score = 50

    mean_yes = mean_affairs.get("yes")
    mean_no = mean_affairs.get("no")
    prop_yes = prop_any.get("yes")
    prop_no = prop_any.get("no")

    if (
        mean_yes is not None
        and mean_no is not None
        and prop_yes is not None
        and prop_no is not None
    ):
        if mean_yes < mean_no and prop_yes < prop_no:
            base_score = min(100, base_score + 5)
        elif mean_yes > mean_no and prop_yes > prop_no:
            base_score = max(0, base_score - 5)

    return int(round(base_score))


def build_explanation(
    dataframe: pd.DataFrame,
    mean_affairs: Dict[str, float],
    prop_any: Dict[str, float],
    effects: Dict[str, Tuple[float, float, float]],
    score: int,
) -> str:
    num_rows = int(len(dataframe))
    mean_yes = mean_affairs.get("yes", float("nan"))
    mean_no = mean_affairs.get("no", float("nan"))
    prop_yes = prop_any.get("yes", float("nan"))
    prop_no = prop_any.get("no", float("nan"))

    unadj_coef, unadj_p, unadj_or = effects["logit_unadjusted"]
    adj_coef, adj_p, adj_or = effects["logit_adjusted"]
    pois_coef, pois_p, pois_rr = effects["poisson"]

    def effect_sentence(
        coef: float, p_value: float, ratio: float, model_label: str, outcome_label: str
    ) -> str:
        if coef < 0:
            direction = "lower"
        elif coef > 0:
            direction = "higher"
        else:
            direction = "no clear change in"

        if p_value < 0.05:
            signif_text = "a statistically significant association"
        else:
            signif_text = "no statistically significant association"

        return (
            f"{model_label}, having children was associated with {direction} {outcome_label} "
            f"(ratio = {ratio:.2f}, p = {p_value:.3f}, {signif_text})."
        )

    if score > 50:
        verdict = "Yes"
        qualitative = "evidence that having children is associated with fewer extramarital affairs"
    elif score < 50:
        verdict = "No"
        qualitative = (
            "little or no evidence that having children decreases extramarital affairs "
            "and the association may even go in the opposite direction"
        )
    else:
        verdict = "Unclear"
        qualitative = (
            "insufficient evidence to determine whether having children decreases extramarital affairs"
        )

    lines = []
    lines.append(
        "Using the provided affairs dataset of "
        f"{num_rows} married individuals, I examined whether having children is associated "
        "with lower engagement in extramarital affairs."
    )
    lines.append(
        "Descriptively, respondents with children reported an average of "
        f"{mean_yes:.2f} affairs versus {mean_no:.2f} among those without children, "
        f"and {prop_yes * 100:.1f}% versus {prop_no * 100:.1f}% had any affairs at all."
    )
    lines.append(
        effect_sentence(
            unadj_coef,
            unadj_p,
            unadj_or,
            "In an unadjusted logistic regression on the indicator of any affair",
            "odds of reporting any affair",
        )
    )
    lines.append(
        effect_sentence(
            adj_coef,
            adj_p,
            adj_or,
            "In a logistic regression adjusting for gender, age, years married, religiousness, education, occupation, and marital rating",
            "odds of reporting any affair",
        )
    )
    lines.append(
        effect_sentence(
            pois_coef,
            pois_p,
            pois_rr,
            "In a Poisson regression on the number of affairs with the same covariates",
            "expected count of affairs",
        )
    )
    lines.append(
        f"Overall, these models provide {qualitative}. I summarize this as a "
        f'\"{verdict}\" answer to the question \"Does having children decrease the engagement in extramarital affairs?\" '
        f"with a response value of {score} on a 0–100 scale (0 = strong \"No\", 100 = strong \"Yes\")."
    )

    return "\n".join(lines)


def main() -> None:
    data_path = Path("affairs.csv")
    dataframe = load_data(data_path)

    mean_affairs, prop_any = compute_group_stats(dataframe)
    effects = fit_models(dataframe)
    score = map_evidence_to_score(effects, mean_affairs, prop_any)
    explanation = build_explanation(dataframe, mean_affairs, prop_any, effects, score)

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as file:
        json.dump(conclusion, file)


if __name__ == "__main__":
    main()

