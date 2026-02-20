import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Restrict to the four genera relevant to the research question.
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Drop any rows with clearly invalid counts.
    df = df[df["sockets"] > 0].copy()
    df = df[df["num_amtl"].between(0, df["sockets"])].copy()

    # Define human vs non-human indicator.
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of antemortem tooth loss for binomial regression.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Center continuous covariates to improve numerical stability.
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Binomial regression with logit link:
    # outcome: proportion of missing teeth, with sockets as binomial trials.
    formula = "prop_amtl ~ human + age_c + prob_male_c + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract the coefficient and p-value for the human indicator.
    coef_human = float(result.params.get("human", np.nan))
    pvalue_human = float(result.pvalues.get("human", np.nan))

    # Compute model-based predicted AMTL probabilities for human vs non-human,
    # averaging over the observed covariate distribution and weighting by sockets.
    design_human = df.copy()
    design_human["human"] = 1
    design_nonhuman = df.copy()
    design_nonhuman["human"] = 0

    pred_human = result.predict(design_human)
    pred_nonhuman = result.predict(design_nonhuman)

    weights = df["sockets"].to_numpy()
    mean_pred_human = float(np.average(pred_human, weights=weights))
    mean_pred_nonhuman = float(np.average(pred_nonhuman, weights=weights))

    # Also compute simple empirical proportions by genus for context.
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(prop=lambda g: g["num_amtl"] / g["sockets"])
    )

    human_prop = float(genus_summary.loc["Homo sapiens", "prop"])
    nonhuman_prop = float(
        genus_summary.loc[["Pan", "Pongo", "Papio"], "num_amtl"].sum()
        / genus_summary.loc[["Pan", "Pongo", "Papio"], "sockets"].sum()
    )

    # Decide Yes/No based on direction and strength of evidence.
    # Require a positive coefficient and p < 0.05 as primary evidence.
    if np.isfinite(coef_human) and coef_human > 0 and pvalue_human < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Map strength of evidence to a confidence score.
    # Start from a baseline and adjust using p-value magnitude.
    if not np.isfinite(coef_human):
        confidence = 50
    else:
        if pvalue_human < 0.001:
            confidence = 90
        elif pvalue_human < 0.01:
            confidence = 85
        elif pvalue_human < 0.05:
            confidence = 75
        elif pvalue_human < 0.1:
            confidence = 65
        else:
            confidence = 55

    # Clip confidence to [0, 100].
    confidence = int(max(0, min(100, round(confidence))))

    explanation_parts = [
        "I fit a binomial regression model with a logit link to the AMTL dataset,",
        "using the proportion of missing teeth (num_amtl / sockets) as the outcome,",
        "and including a human vs non-human indicator, age, sex (probability of being male),",
        "and tooth class as predictors, while weighting each row by the number of sockets.",
        f"The estimated coefficient for the human indicator was {coef_human:.3f} with p-value {pvalue_human:.3g},",
        "indicating the direction and statistical strength of the difference in AMTL frequency between humans and non-human primates after adjustment.",
        f"Model-based predicted AMTL probabilities, averaged over the observed covariate distribution, were approximately {mean_pred_human:.3%} for humans",
        f"and {mean_pred_nonhuman:.3%} for non-human primates.",
        f"Empirically, the overall AMTL proportion was {human_prop:.3%} for Homo sapiens and {nonhuman_prop:.3%} when pooling Pan, Pongo, and Papio.",
        "Based on the sign and significance of the human coefficient together with these adjusted and unadjusted comparisons,",
        f"I conclude that the answer to the research question is '{response}'.",
    ]

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

