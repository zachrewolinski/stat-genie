import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Restrict to the four genera of interest.
    genera_of_interest = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Construct variables for the binomial model.
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Fit a binomial GLM for AMTL proportion, accounting for age, sex (prob_male),
    # and tooth class, with sockets as the binomial trial count.
    model = smf.glm(
        formula="prop_amtl ~ human + age + prob_male + C(tooth_class)",
        data=df,
        family=Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Extract effect of being human vs non-human primate.
    coef_human = float(model.params["human"])
    se_human = float(model.bse["human"])
    pval_human = float(model.pvalues["human"])
    odds_ratio = float(np.exp(coef_human))

    # Compute average predicted AMTL probabilities per socket for humans vs non-humans,
    # using the fitted model while holding other covariates at their observed values.
    df_nonhuman = df.copy()
    df_nonhuman["human"] = 0
    df_human = df.copy()
    df_human["human"] = 1

    pred_nonhuman = model.predict(df_nonhuman)
    pred_human = model.predict(df_human)

    avg_nonhuman = float(np.average(pred_nonhuman, weights=df["sockets"]))
    avg_human = float(np.average(pred_human, weights=df["sockets"]))
    abs_diff = avg_human - avg_nonhuman
    rel_ratio = avg_human / avg_nonhuman if avg_nonhuman > 0 else float("inf")

    # Determine binary answer, strength, and confidence based on effect size and p-value.
    if coef_human > 0:
        response = "Yes"
    else:
        response = "No"

    # Map statistical evidence to a 0–100 strength scale.
    # Start from a baseline tied to standardized effect size.
    z_score = coef_human / se_human if se_human > 0 else 0.0
    effect_strength = min(1.0, abs(z_score) / 5.0)

    # Incorporate p-value: smaller p-values indicate stronger evidence.
    if pval_human <= 1e-6:
        p_component = 1.0
    elif pval_human <= 0.001:
        p_component = 0.9
    elif pval_human <= 0.01:
        p_component = 0.75
    elif pval_human <= 0.05:
        p_component = 0.6
    elif pval_human <= 0.1:
        p_component = 0.45
    else:
        p_component = 0.25

    strength = int(round(100 * 0.5 * (effect_strength + p_component)))

    # Confidence reflects data volume and model clarity; here we tie it mainly to p-value.
    if pval_human <= 1e-6:
        confidence = 95
    elif pval_human <= 0.001:
        confidence = 90
    elif pval_human <= 0.01:
        confidence = 85
    elif pval_human <= 0.05:
        confidence = 80
    elif pval_human <= 0.1:
        confidence = 70
    else:
        confidence = 55

    explanation = {
        "model_summary": {
            "coef_human": coef_human,
            "se_human": se_human,
            "pval_human": pval_human,
            "odds_ratio_human": odds_ratio,
        },
        "predicted_amtl": {
            "avg_prob_nonhuman": avg_nonhuman,
            "avg_prob_human": avg_human,
            "absolute_difference": abs_diff,
            "relative_ratio_human_over_nonhuman": rel_ratio,
        },
        "reasoning": (
            "I fit a binomial regression model for the proportion of antemortem tooth "
            "loss (num_amtl out of sockets) using a logit link and sockets as the "
            "binomial trial count. The predictors were an indicator for modern humans "
            "(Homo sapiens vs. Pan/Pongo/Papio), age at death, probability of being "
            "male (prob_male), and tooth class (Anterior/Posterior/Premolar). "
            "The coefficient for the human indicator was translated into an odds ratio "
            "and evaluated with its standard error and p-value, and I compared the "
            "model-predicted AMTL probabilities per socket for hypothetical human and "
            "non-human specimens while holding age, sex, and tooth class at their "
            "observed distributions. The response, strength, and confidence values are "
            "based on the sign and magnitude of the human coefficient, its p-value, "
            "and the size of the predicted difference in AMTL frequencies."
        ),
    }

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": json.dumps(explanation),
    }

    # Write the required JSON object to conclusion.txt with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

