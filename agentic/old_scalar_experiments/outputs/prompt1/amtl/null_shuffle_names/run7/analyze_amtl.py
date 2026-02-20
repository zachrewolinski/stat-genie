import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Map anonymized column names to their substantive meanings
    # genus      -> number of missing teeth (AMTL count)
    # age        -> number of observable sockets
    # pop        -> estimated age at death
    # stdev_age  -> sex estimate (probability of male)
    # sockets    -> tooth class (Anterior / Posterior / Premolar)
    # tooth_class-> taxonomic genus label (e.g., Homo sapiens, Pan, Papio, Pongo)

    df["n_missing"] = df["genus"].astype(float)
    df["n_sockets"] = df["age"].astype(float)

    # Keep only rows with a positive number of observable sockets
    df = df[df["n_sockets"] > 0].copy()

    # Proportion of teeth missing for each specimen / tooth-class combination
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Indicator for modern humans vs. non-human primates
    df["genus_label"] = df["tooth_class"].astype(str)
    df["human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Covariates: age at death, sex estimate, and tooth class
    df["age_death"] = df["pop"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)
    df["tooth_class_cat"] = df["sockets"].astype("category")

    # Drop any rows with missing covariates
    df = df.dropna(
        subset=["prop_missing", "human", "age_death", "sex_prob_male", "tooth_class_cat"]
    )

    # Fit a binomial regression (logit link) on the AMTL proportion,
    # using the number of observable sockets as frequency weights so
    # that each row represents multiple Bernoulli trials.
    model = smf.glm(
        formula="prop_missing ~ human + age_death + sex_prob_male + C(tooth_class_cat)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    # Extract effect of being human vs. non-human primate
    coef_human = float(result.params["human"])
    se_human = float(result.bse["human"])
    p_human = float(result.pvalues["human"])
    or_human = float(np.exp(coef_human))
    ci_low = float(np.exp(coef_human - 1.96 * se_human))
    ci_high = float(np.exp(coef_human + 1.96 * se_human))

    # Simple descriptive comparison of AMTL proportions by genus
    def _mean_prop(group: pd.DataFrame) -> float:
        return float(group["n_missing"].sum() / group["n_sockets"].sum())

    group_stats = {
        genus_name: {"mean_prop_missing": _mean_prop(group)}
        for genus_name, group in df.groupby("genus_label")
    }

    # Decide binary answer based on sign and significance of the human effect
    if coef_human > 0 and p_human < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string summarizing modeling approach and key results
    explanation_parts = []
    explanation_parts.append(
        "We modeled the probability of antemortem tooth loss (AMTL) using a binomial "
        "regression where the outcome was the number of missing teeth out of the "
        "observable sockets for each specimen and tooth class. The model included "
        "predictors for whether the specimen was a modern human (Homo sapiens) versus "
        "a non-human primate, estimated age at death, sex estimate (probability of "
        "male), and tooth class (anterior, posterior, premolar)."
    )

    explanation_parts.append(
        f"In this model, the coefficient for the human indicator was {coef_human:.3f}, "
        f"corresponding to an odds ratio of {or_human:.2f} with a 95% confidence "
        f"interval from {ci_low:.2f} to {ci_high:.2f} (p = {p_human:.3g})."
    )

    # Add concise descriptive statistics by genus
    desc_segments = []
    for genus_name, stats in group_stats.items():
        desc_segments.append(
            f"{genus_name}: mean AMTL proportion ≈ {stats['mean_prop_missing']:.3f}"
        )
    if desc_segments:
        explanation_parts.append(
            "Descriptively, the overall proportion of missing teeth by genus was: "
            + "; ".join(desc_segments)
            + "."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the human coefficient is positive and statistically significant "
            "after controlling for age, sex, and tooth class, this analysis indicates "
            "that modern humans have higher AMTL frequencies than the non-human primate "
            "genera in this sample."
        )
    else:
        explanation_parts.append(
            "Because the human coefficient is not both positive and statistically "
            "significant after controlling for age, sex, and tooth class, this analysis "
            "does not provide strong evidence that modern humans have higher AMTL "
            "frequencies than the non-human primate genera in this sample."
        )

    explanation = " ".join(explanation_parts)

    output = {"response": response, "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()

