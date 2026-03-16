import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity filters: keep rows with positive socket counts
    df = df[df["sockets"] > 0].copy()

    # Drop clearly invalid rows where missing teeth exceed observable sockets
    df = df[df["num_amtl"] <= df["sockets"]].copy()

    # Create a binary indicator for humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Use aggregated binomial response with frequency weights (sockets)
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # C(tooth_class) and age, prob_male as covariates
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def evaluate_human_effect(result) -> dict:
    # Extract coefficient and p-value for human indicator
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    # Compute average predicted AMTL probabilities for humans vs non-humans
    # using the observed covariate distribution (marginal effects).
    # This is more interpretable than raw coefficients.
    df = result.model.data.frame.copy()

    df_human = df.copy()
    df_human["is_human"] = 1
    human_pred = result.predict(df_human).mean()

    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    nonhuman_pred = result.predict(df_nonhuman).mean()

    diff = human_pred - nonhuman_pred

    return {
        "coef": float(coef),
        "pval": float(pval),
        "human_pred": float(human_pred),
        "nonhuman_pred": float(nonhuman_pred),
        "diff": float(diff),
    }


def map_to_likert(human_effect: dict) -> int:
    """
    Map the evidence to a 0-100 Likert scale where higher values
    correspond to stronger evidence that humans have higher AMTL
    than non-human primates, after adjusting for covariates.

    We combine the sign and magnitude of the effect and its p-value:
    - Strong, significant positive effect (p < 0.01): 80-100
    - Moderate positive effect (0.01 <= p < 0.05): 60-80
    - Weak or non-significant effect: around 50
    - Significant negative effect: below 50
    """
    coef = human_effect["coef"]
    pval = human_effect["pval"]
    diff = human_effect["diff"]

    # If the direction is opposite (humans *lower* AMTL), we move below 50
    if pval < 0.01:
        if diff > 0:
            score = 90
        elif diff < 0:
            score = 10
        else:
            score = 50
    elif pval < 0.05:
        if diff > 0:
            score = 70
        elif diff < 0:
            score = 30
        else:
            score = 50
    elif pval < 0.1:
        if diff > 0:
            score = 60
        elif diff < 0:
            score = 40
        else:
            score = 50
    else:
        # No strong evidence either way; anchor near uncertainty
        # but nudge slightly based on effect direction and magnitude.
        if diff > 0:
            score = 55
        elif diff < 0:
            score = 45
        else:
            score = 50

    # Ensure integer 0-100
    score = int(round(min(max(score, 0), 100)))
    return score


def build_explanation(human_effect: dict, likert_score: int) -> str:
    coef = human_effect["coef"]
    pval = human_effect["pval"]
    human_pred = human_effect["human_pred"]
    nonhuman_pred = human_effect["nonhuman_pred"]
    diff = human_effect["diff"]

    if likert_score > 50:
        direction_sentence = (
            "The regression results suggest that, after controlling for age, "
            "sex (probability of being male), and tooth class, modern humans "
            "have *higher* predicted frequencies of antemortem tooth loss "
            "than non-human primates."
        )
    elif likert_score < 50:
        direction_sentence = (
            "The regression results suggest that, after controlling for age, "
            "sex (probability of being male), and tooth class, modern humans "
            "have *lower* predicted frequencies of antemortem tooth loss "
            "than non-human primates."
        )
    else:
        direction_sentence = (
            "The regression results do not provide clear evidence that "
            "modern humans differ from non-human primates in their "
            "frequencies of antemortem tooth loss after controlling for "
            "age, sex (probability of being male), and tooth class."
        )

    explanation = (
        "I analyzed the AMTL dataset using a binomial regression model where the "
        "response was the proportion of missing teeth out of observable sockets "
        "for each specimen and tooth class. The model included a binary indicator "
        "for modern humans (Homo sapiens vs. Pan, Pongo, Papio), age at death, "
        "probability of being male, and categorical tooth class as predictors.\n\n"
        f"{direction_sentence}\n\n"
        f"In the fitted model, the coefficient for the human indicator was "
        f"{coef:.3f} with a p-value of {pval:.3g}. Using the observed covariate "
        "distribution, the average predicted AMTL probability for humans was "
        f"{human_pred:.3%}, compared to {nonhuman_pred:.3%} for non-human primates "
        f"(an absolute difference of {diff:.3%}). "
        "This effect size and its statistical significance were used to map the "
        "answer onto a 0–100 Likert scale, where 0 represents a strong 'No' to "
        "the question 'Do modern humans have higher AMTL than non-human primates?' "
        "and 100 represents a strong 'Yes'. "
        f"The resulting score of {likert_score} reflects the strength and "
        "direction of the evidence from this model."
    )

    return explanation


def main():
    base = Path(__file__).parent
    df = load_data(base / "amtl.csv")
    result = fit_binomial_model(df)
    human_effect = evaluate_human_effect(result)
    likert_score = map_to_likert(human_effect)
    explanation = build_explanation(human_effect, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}

    out_path = base / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
