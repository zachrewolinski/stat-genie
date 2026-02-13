import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Reconstruct variable semantics from metadata:
    # - sockets: tooth class (Anterior / Posterior / Premolar)
    # - tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
    # - genus: number of missing teeth in that class
    # - age: number of observable sockets (trials)
    # - pop: estimated age at death
    # - stdev_age: estimate of sex (probability male, coded 0–1 in 0.25 steps)
    df["tooth_class_cat"] = df["sockets"]
    df["genus_cat"] = df["tooth_class"]
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male_est"] = df["stdev_age"].astype(float)

    # Exclude any rows with zero observable sockets to avoid invalid binomial trials
    df = df[df["num_sockets"] > 0].copy()

    # Group indicator: modern humans vs non-human primates
    df["is_human"] = (df["genus_cat"] == "Homo sapiens").astype(int)

    # Response for binomial regression: proportion missing with number of trials as weights
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]

    # Ensure categorical encoding
    df["tooth_class_cat"] = df["tooth_class_cat"].astype("category")

    # Fit binomial GLM with logit link
    # missing_prop ~ is_human + age_at_death + prob_male_est + tooth_class
    model = smf.glm(
        formula="missing_prop ~ is_human + age_at_death + prob_male_est + C(tooth_class_cat)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )

    result = model.fit()

    # Extract human effect
    human_coef = result.params.get("is_human", np.nan)
    human_pval = result.pvalues.get("is_human", np.nan)
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else np.nan

    # Predicted AMTL probabilities for a representative individual
    mean_age = float(df["age_at_death"].mean())
    mean_prob_male = float(df["prob_male_est"].mean())
    ref_tooth_class = df["tooth_class_cat"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [1, 0],
            "age_at_death": [mean_age, mean_age],
            "prob_male_est": [mean_prob_male, mean_prob_male],
            "tooth_class_cat": [ref_tooth_class, ref_tooth_class],
        }
    )

    pred = result.get_prediction(pred_df).summary_frame(alpha=0.05)
    p_human = float(pred.loc[0, "mean"])
    p_nonhuman = float(pred.loc[1, "mean"])

    # Decision rule:
    # Answer "Yes" only if humans have higher estimated AMTL probability
    # AND the is_human coefficient is significantly > 0 at alpha = 0.05.
    has_higher_point_estimate = p_human > p_nonhuman
    is_significant = (human_coef > 0) and (human_pval < 0.05)

    response = "Yes" if has_higher_point_estimate and is_significant else "No"

    # Build explanation string
    explanation = (
        "I fit a binomial logistic regression of the proportion of missing teeth "
        "(number missing out of observable sockets) on an indicator for modern humans "
        "versus non-human primates (Pan, Papio, Pongo), while adjusting for age at death, "
        "estimated probability of being male, and tooth class (anterior, posterior, premolar). "
        f"The estimated log-odds coefficient for modern humans was {human_coef:.3f}, "
        f"corresponding to an odds ratio of {human_or:.2f} (p-value = {human_pval:.3g}). "
        f"For an individual with average age and sex estimate and a typical tooth class "
        f"({ref_tooth_class}), the model-predicted AMTL probability was "
        f"{p_human:.3f} for modern humans and {p_nonhuman:.3f} for non-human primates. "
        "Based on this model, I concluded that modern humans "
        + ("do" if response == "Yes" else "do not")
        + " have significantly higher frequencies of antemortem tooth loss after accounting "
        "for age, sex, and tooth class."
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

