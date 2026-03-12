import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    # Exclude rows with zero sockets to avoid division by zero
    df = df[df["sockets"] > 0].copy()

    # Create proportion missing and human indicator
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial GLM modeling AMTL proportion as a function of human vs non-human,
    controlling for age, sex (prob_male), and tooth class.
    """
    # Use proportion response with sockets as weights
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    # Extract coefficient and p-value for is_human
    params = result.params
    pvalues = result.pvalues

    human_coef = params.get("is_human", np.nan)
    human_p = pvalues.get("is_human", np.nan)

    # Compute naive genus-level AMTL proportions for context
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .assign(prop=lambda x: x["total_missing"] / x["total_sockets"])
        .reset_index()
    )

    human_prop = float(
        genus_summary.loc[genus_summary["genus"] == "Homo sapiens", "prop"].iloc[0]
    )
    nonhuman_prop = float(
        genus_summary.loc[genus_summary["genus"] != "Homo sapiens", "total_missing"].sum()
        / genus_summary.loc[genus_summary["genus"] != "Homo sapiens", "total_sockets"].sum()
    )

    # Decide Likert scale response based on sign and significance
    # Strong evidence humans have higher AMTL: positive coef and p < 0.01
    # Moderate evidence: positive coef and 0.01 <= p < 0.05
    # Weak or no evidence: p >= 0.05 (lean toward "No")
    if np.isnan(human_coef) or np.isnan(human_p):
        likert = 50
        conclusion = (
            "Model could not estimate a distinct human effect on AMTL; "
            "evidence is inconclusive regarding higher AMTL in modern humans."
        )
    else:
        if human_p < 0.01 and human_coef > 0:
            likert = 85
            conclusion = (
                "There is strong statistical evidence that modern humans have higher "
                "antemortem tooth loss (AMTL) frequencies than non-human primates "
                "after adjusting for age, sex, and tooth class."
            )
        elif human_p < 0.05 and human_coef > 0:
            likert = 70
            conclusion = (
                "There is moderate statistical evidence that modern humans have higher "
                "AMTL frequencies than non-human primates after accounting for age, "
                "sex, and tooth class."
            )
        elif human_p < 0.05 and human_coef < 0:
            likert = 20
            conclusion = (
                "There is statistically significant evidence that modern humans have "
                "lower AMTL frequencies than non-human primates after adjusting for "
                "age, sex, and tooth class."
            )
        elif human_p < 0.1 and human_coef > 0:
            likert = 60
            conclusion = (
                "There is weak statistical evidence that modern humans may have higher "
                "AMTL frequencies than non-human primates after accounting for age, "
                "sex, and tooth class, but the result is only marginally significant."
            )
        elif human_p < 0.1 and human_coef < 0:
            likert = 30
            conclusion = (
                "There is weak statistical evidence that modern humans may have lower "
                "AMTL frequencies than non-human primates after accounting for age, "
                "sex, and tooth class, but the result is only marginally significant."
            )
        else:
            # p >= 0.1: treat as no convincing evidence for a difference
            likert = 40
            conclusion = (
                "After adjusting for age, sex, and tooth class, there is no convincing "
                "statistical evidence that modern humans have higher AMTL frequencies "
                "than non-human primates."
            )

    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) exhibit higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?",
        "",
        "Modeling approach:",
        "- Fitted a binomial generalized linear model with logit link using tooth-level "
        "proportions of missing teeth as the response (num_amtl / sockets).",
        "- Included a binary indicator for modern humans versus non-human primates, and "
        "controlled for age (continuous), sex (probability of being male), and tooth class "
        "as a categorical predictor.",
        "- Used the number of observable tooth sockets as frequency weights to reflect "
        "the differing number of trials per observation.",
        "",
        f"Key results:",
        f"- Naive AMTL proportion for modern humans: {human_prop:.3f}.",
        f"- Naive AMTL proportion for non-human primates (combined): {nonhuman_prop:.3f}.",
        f"- Regression coefficient for the human indicator (log-odds scale): {human_coef:.3f}.",
        f"- p-value for the human indicator: {human_p:.3g}.",
        "",
        "Interpretation:",
        conclusion,
    ]

    return {
        "likert": int(round(likert)),
        "explanation": "\n".join(explanation_lines),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    df = load_data(base_dir / "amtl.csv")
    result = fit_model(df)
    summary = summarize_effect(df, result)

    output = {
        "response": int(summary["likert"]),
        "explanation": summary["explanation"],
    }

    with open(base_dir / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

