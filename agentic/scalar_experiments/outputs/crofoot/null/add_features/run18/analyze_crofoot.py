import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only the columns relevant to the research question
    cols = ["win", "n_focal", "n_other", "dist_focal", "dist_other"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in data: {missing}")
    return df[cols].copy()


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Relative group size as log ratio of focal to other
    df = df.copy()
    # Avoid division by zero; according to metadata n_focal/n_other are >= 5
    df["rel_size_log"] = np.log(df["n_focal"] / df["n_other"])
    # Contest location advantage: positive when focal is closer to its home range center
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]
    return df


def fit_logistic(df: pd.DataFrame):
    y = df["win"]
    X = df[["rel_size_log", "loc_adv"]]
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def summarize_results(result) -> dict:
    params = result.params
    pvalues = result.pvalues

    # Odds ratios
    odds_ratios = np.exp(params)

    summary = {
        "coefficients": params.to_dict(),
        "pvalues": pvalues.to_dict(),
        "odds_ratios": odds_ratios.to_dict(),
    }
    return summary


def derive_likert_response(summary: dict) -> tuple[int, str]:
    p_rel = summary["pvalues"]["rel_size_log"]
    p_loc = summary["pvalues"]["loc_adv"]
    beta_rel = summary["coefficients"]["rel_size_log"]
    beta_loc = summary["coefficients"]["loc_adv"]
    or_rel = summary["odds_ratios"]["rel_size_log"]
    or_loc = summary["odds_ratios"]["loc_adv"]

    explanations = []

    # Interpret relative group size
    if p_rel < 0.05:
        explanations.append(
            f"Relative group size has a statistically significant effect on winning (p = {p_rel:.3f}), "
            f"with larger focal groups more likely to win (odds ratio ≈ {or_rel:.2f})."
            if beta_rel > 0
            else f"Relative group size has a statistically significant but negative association with winning (p = {p_rel:.3f}, odds ratio ≈ {or_rel:.2f})."
        )
    else:
        explanations.append(
            f"Relative group size does not show strong statistical evidence of influencing win probability (p = {p_rel:.3f}, odds ratio ≈ {or_rel:.2f})."
        )

    # Interpret location advantage
    if p_loc < 0.05:
        direction = "closer to its home range center" if beta_loc > 0 else "farther from its home range center"
        explanations.append(
            f"Contest location also has a statistically significant effect (p = {p_loc:.3f}); "
            f"when the focal group is {direction} relative to the other group, its odds of winning change (odds ratio per unit of location advantage ≈ {or_loc:.2f})."
        )
    else:
        explanations.append(
            f"Contest location (relative distance to home range centers) does not provide strong statistical evidence of affecting win probability (p = {p_loc:.3f}, odds ratio ≈ {or_loc:.2f})."
        )

    # Map statistical evidence to Likert response
    # Strong evidence for both predictors -> high score
    if p_rel < 0.01 and p_loc < 0.01:
        score = 90
    # Strong evidence for one and moderate for the other
    elif (p_rel < 0.01 and p_loc < 0.05) or (p_loc < 0.01 and p_rel < 0.05):
        score = 80
    # Both significant at 5% but not 1%
    elif p_rel < 0.05 and p_loc < 0.05:
        score = 75
    # Only one significant at 5%
    elif (p_rel < 0.05) != (p_loc < 0.05):
        score = 60
    # Neither significant but trends in expected directions
    elif beta_rel > 0 and beta_loc > 0:
        score = 40
    else:
        score = 25

    explanation_text = " ".join(explanations)
    return int(score), explanation_text


def main():
    df = load_data(Path("crofoot.csv"))
    df = add_derived_variables(df)
    result = fit_logistic(df)
    summary = summarize_results(result)
    score, explanation = derive_likert_response(summary)

    conclusion = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

