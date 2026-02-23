import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Rename columns to reflect their semantic meaning based on the metadata
    df = df.rename(
        columns={
            "sockets": "tooth_class",      # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",    # specimen identifier
            "genus": "num_missing",        # number of missing teeth in this class
            "age": "num_sockets",          # number of observable sockets
            "pop": "age_at_death",         # estimated age at death
            "num_amtl": "age_at_death_sd", # uncertainty in age at death
            "stdev_age": "prob_male",      # probability specimen is male
            "tooth_class": "genus",        # taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "specimen": "population",      # population / region
        }
    )

    # Basic data integrity filters
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0)].copy()
    df = df[df["num_missing"] <= df["num_sockets"]].copy()

    # Keep only the four genera of interest
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])].copy()

    # Human indicator (1 = modern human, 0 = non-human primate)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Aggregated-binomial response: proportion missing with number of sockets as weights
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    model_df = df[
        [
            "prop_missing",
            "is_human",
            "age_at_death",
            "prob_male",
            "tooth_class",
            "num_sockets",
        ]
    ].dropna()

    n_rows = int(model_df.shape[0])

    # Most common tooth class for predictions
    common_tooth_class = model_df["tooth_class"].mode().iat[0]

    # Fit binomial GLM: proportion missing ~ human indicator + age + sex + tooth class
    glm = smf.glm(
        "prop_missing ~ is_human + age_at_death + prob_male + C(tooth_class)",
        data=model_df,
        family=sm.families.Binomial(),
        freq_weights=model_df["num_sockets"],
    )
    result = glm.fit()

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef))

    # Predicted probabilities for typical human vs non-human at average covariates
    mean_age = float(model_df["age_at_death"].mean())
    mean_prob_male = float(model_df["prob_male"].mean())

    pred_df = pd.DataFrame(
        {
            "is_human": [1, 0],
            "age_at_death": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )
    pred_probs = result.predict(pred_df)
    p_human = float(pred_probs.iloc[0])
    p_nonhuman = float(pred_probs.iloc[1])

    # Map effect and significance to a 0–100 Likert score
    if coef > 0:
        if pval < 0.001:
            likert = 95
        elif pval < 0.01:
            likert = 85
        elif pval < 0.05:
            likert = 75
        elif pval < 0.1:
            likert = 65
        else:
            likert = 55
    elif coef < 0:
        if pval < 0.001:
            likert = 5
        elif pval < 0.01:
            likert = 15
        elif pval < 0.05:
            likert = 25
        elif pval < 0.1:
            likert = 35
        else:
            likert = 45
    else:
        likert = 50

    # Qualitative interpretation for the explanation text
    if coef > 0 and pval < 0.05:
        conclusion_phrase = (
            "These results indicate that modern humans have higher frequencies of "
            "antemortem tooth loss than non-human primates after accounting for age, "
            "sex, and tooth class."
        )
    elif coef < 0 and pval < 0.05:
        conclusion_phrase = (
            "These results indicate that modern humans have lower frequencies of "
            "antemortem tooth loss than non-human primates after accounting for age, "
            "sex, and tooth class."
        )
    else:
        if pval < 0.1:
            strength = "only weakly"
        else:
            strength = "not"
        conclusion_phrase = (
            f"Because the human indicator is {strength} statistically significant "
            "after controlling for age, sex, and tooth class, the data do not provide "
            "strong evidence that modern humans differ from non-human primates in "
            "their frequencies of antemortem tooth loss."
        )

    explanation_parts = []
    explanation_parts.append(
        f"Using binomial regression on {n_rows} tooth-class-by-individual observations, "
        "I modeled the proportion of antemortem tooth loss (number of missing teeth "
        "divided by observable sockets) as a function of a human-versus-non-human "
        "indicator, estimated age at death, probability of being male, and tooth "
        "class (anterior, premolar, posterior)."
    )
    explanation_parts.append(
        f" The estimated coefficient for the human indicator is {coef:.3f} "
        f"(odds ratio {or_human:.2f}, p-value {pval:.3g}), while the model-predicted "
        f"proportion of missing teeth at average covariate values is {p_human:.3f} "
        f"for humans versus {p_nonhuman:.3f} for non-human primates."
    )
    explanation_parts.append(f" {conclusion_phrase}")

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(likert), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

