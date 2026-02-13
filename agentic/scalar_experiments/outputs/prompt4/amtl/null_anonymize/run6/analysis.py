import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Prepare variables
    df = df.copy()
    df["missing"] = df["feature3"]
    df["total"] = df["feature4"]
    # Exclude any rows with non-positive totals just in case
    df = df[df["total"] > 0].reset_index(drop=True)

    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)
    df["age"] = df["feature5"]
    df["sex_est"] = df["feature7"]
    df["tooth_class"] = df["feature1"]

    # Basic descriptive statistics for explanation
    n_specimens = len(df)
    n_human = int(df["is_human"].sum())
    n_nonhuman = int(n_specimens - n_human)

    human_mask = df["is_human"] == 1
    human_missing = float(df.loc[human_mask, "missing"].sum())
    human_total = float(df.loc[human_mask, "total"].sum())
    non_missing = float(df.loc[~human_mask, "missing"].sum())
    non_total = float(df.loc[~human_mask, "total"].sum())

    human_rate = human_missing / human_total if human_total > 0 else np.nan
    non_rate = non_missing / non_total if non_total > 0 else np.nan

    # Model the proportion of missing teeth using a logit transformation.
    # A small continuity correction is applied so that 0% and 100% values
    # remain finite on the logit scale, and we then use linear regression.
    df["prop_missing"] = df["missing"] / df["total"]
    df["prop_adj"] = (df["missing"] + 0.5) / (df["total"] + 1.0)
    df["logit_prop"] = np.log(df["prop_adj"] / (1.0 - df["prop_adj"]))

    model = smf.ols(
        formula="logit_prop ~ is_human + age + sex_est + C(tooth_class)",
        data=df,
    )
    result = model.fit()

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    conf_int = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(conf_int[0]))
    or_ci_high = float(np.exp(conf_int[1]))

    # Decide on Yes/No answer to the research question:
    # "Do modern humans have higher AMTL frequencies than non-human primates,
    # after accounting for age, sex, and tooth class?"
    # We only answer "Yes" when the human coefficient is positive and statistically
    # significant at the 0.05 level; otherwise we answer "No".
    if coef > 0 and pval < 0.05:
        answer = "yes"
    else:
        answer = "no"

    # Map evidence into a 0–100 Likert-style score where
    # 0 = strong "No" and 100 = strong "Yes".
    response = likelihood_score(answer=answer, coef=coef, pval=pval, odds_ratio=odds_ratio)

    # Build a concise explanation grounded in the model output
    explanation = build_explanation(
        n_specimens=n_specimens,
        n_human=n_human,
        n_nonhuman=n_nonhuman,
        human_rate=human_rate,
        non_rate=non_rate,
        coef=coef,
        pval=pval,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        answer=answer,
        response=response,
    )

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def likelihood_score(answer: str, coef: float, pval: float, odds_ratio: float) -> int:
    """
    Convert model evidence into a scalar on [0, 100].

    - Values > 50 indicate a "Yes" answer (humans have higher AMTL).
    - Values < 50 indicate a "No" answer.
    - Values near 50 indicate very weak evidence either way.
    """
    # Base strength from p-value
    if pval < 0.001:
        strength = 1.0
    elif pval < 0.01:
        strength = 0.9
    elif pval < 0.05:
        strength = 0.8
    elif pval < 0.1:
        strength = 0.6
    elif pval < 0.2:
        strength = 0.5
    elif pval < 0.5:
        strength = 0.4
    else:
        strength = 0.2

    # Adjust strength based on effect size (distance of odds ratio from 1.0)
    effect = abs(np.log(odds_ratio)) if odds_ratio > 0 else 0.0
    if effect < 0.1:  # odds ratio very close to 1
        strength *= 0.5
    elif effect < 0.25:
        strength *= 0.8
    else:
        strength *= 1.0

    # Map to 0–100 scale; keep a neutral zone near p ~ 1 and OR ~ 1
    if answer == "yes":
        score = 50 + strength * 50
    else:
        score = 50 - strength * 50

    score = max(0.0, min(100.0, score))
    return int(round(score))


def build_explanation(
    n_specimens: int,
    n_human: int,
    n_nonhuman: int,
    human_rate: float,
    non_rate: float,
    coef: float,
    pval: float,
    odds_ratio: float,
    or_ci_low: float,
    or_ci_high: float,
    answer: str,
    response: int,
) -> str:
    """Create a concise narrative explaining the analysis and conclusion."""

    human_rate_pct = human_rate * 100 if np.isfinite(human_rate) else float("nan")
    non_rate_pct = non_rate * 100 if np.isfinite(non_rate) else float("nan")

    if answer == "yes":
        qualitative = (
            "These results indicate that, after adjusting for age, sex, and tooth class, "
            "modern humans have higher odds of antemortem tooth loss than the combined "
            "group of non-human primates."
        )
    else:
        if coef < 0 and pval < 0.05:
            qualitative = (
                "These results indicate that, after adjusting for age, sex, and tooth class, "
                "modern humans actually have lower odds of antemortem tooth loss than the "
                "combined group of non-human primates."
            )
        elif coef < 0:
            qualitative = (
                "The human coefficient is negative, suggesting lower AMTL odds in humans, "
                "but the evidence is statistically weak."
            )
        else:
            qualitative = (
                "The human coefficient is positive but not statistically significant, so "
                "there is no strong evidence that humans have higher AMTL odds than "
                "non-human primates after adjustment."
            )

    explanation = (
        f"I analyzed the AMTL dataset (n={n_specimens} tooth-class specimens: "
        f"{n_human} humans and {n_nonhuman} non-human primates) by modeling the "
        f"logit-transformed proportion of missing teeth (number missing out of observable "
        f"sockets, with a small continuity correction) as a function of human vs "
        f"non-human status, age at death, sex estimate, and tooth class (anterior, "
        f"posterior, premolar) using linear regression on the logit scale, which "
        f"approximates a binomial logistic regression at the specimen level. "
        f"Raw missing-tooth proportions were approximately "
        f"{human_rate_pct:.1f}% for humans and {non_rate_pct:.1f}% for non-human primates. "
        f"In the regression model, the human indicator had a log-odds coefficient of "
        f"{coef:.3f} (odds ratio {odds_ratio:.2f}, 95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, "
        f"p-value {pval:.3g}). "
        f"{qualitative} "
        f"The overall strength of evidence is summarized by the Likert-style response "
        f"score of {response}, where values near 0 represent a strong 'No' answer and "
        f"values near 100 represent a strong 'Yes' answer to the question of whether "
        f"humans have higher AMTL frequencies than non-human primates."
    )

    return explanation


if __name__ == "__main__":
    main()
