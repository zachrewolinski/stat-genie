import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # According to info.json metadata, column *names* and semantic descriptions
    # are slightly misaligned. Here we explicitly map columns to their intended meaning.
    #
    # - sockets (values: Anterior/Posterior/Premolar) -> tooth class
    # - tooth_class (values: Homo sapiens, Pan, Papio, Pongo) -> genus / species group
    # - genus (numeric) -> number of missing teeth of that class
    # - age (numeric) -> number of observable sockets
    # - pop (numeric) -> estimated age at death
    # - stdev_age (0–1) -> estimate / probability of being male
    #
    # We keep the original columns but create clearly named analysis variables.

    # Counts for binomial model
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)

    # Guard against any malformed rows
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0)]

    # Predictors
    df["species"] = df["tooth_class"].astype(str)
    df["is_human"] = df["species"].str.contains("Homo", case=False, na=False).astype(int)

    df["tooth_class_cat"] = df["sockets"].astype(str)
    df["age_years"] = df["pop"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)

    # Response as proportion with binomial weights
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]
    # Clip extreme proportions very slightly to avoid numerical issues at 0 or 1
    df["prop_missing"] = df["prop_missing"].clip(1e-6, 1 - 1e-6)

    # Design matrix: is_human (key effect), plus age, sex, and tooth class dummies
    X = pd.get_dummies(
        df[["is_human", "age_years", "sex_prob_male", "tooth_class_cat"]],
        columns=["tooth_class_cat"],
        drop_first=True,
    )
    X = sm.add_constant(X)

    y = df["prop_missing"].to_numpy()
    weights = df["num_sockets"].to_numpy()

    # Fit binomial GLM with logit link, using sockets as binomial trials
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    # Extract key effect for modern humans vs non-human primates
    human_coef = float(result.params["is_human"])
    human_pval = float(result.pvalues["is_human"])

    # Summarize the magnitude of the human effect:
    # odds ratio > 1 means higher AMTL frequency for humans, < 1 means lower.
    human_odds_ratio = float(np.exp(human_coef))

    # Decide binary response based on sign and statistical significance
    alpha = 0.05
    if human_coef > 0 and human_pval < alpha:
        response = "Yes"
    else:
        response = "No"

    # Build a concise explanation
    explanation_parts = [
        "I fit a binomial regression model for the proportion of missing teeth "
        "(number of missing teeth out of observable sockets) using the AMTL dataset.",
        "The model included a binary indicator for modern humans (Homo) versus non-human primates (Pan, Papio, Pongo), "
        "and controlled for estimated age at death, sex (probability of being male), and tooth class (anterior, posterior, premolar).",
        f"The estimated coefficient for the human indicator was {human_coef:.3f}, corresponding to an odds ratio of "
        f"{human_odds_ratio:.2f} for antemortem tooth loss in humans relative to non-human primates, holding age, sex, "
        "and tooth class constant.",
        f"The associated p-value for this coefficient was {human_pval:.3g}.",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Because the human coefficient is positive and statistically significant at the 0.05 level, "
            "this indicates that modern humans have higher frequencies of antemortem tooth loss than the non-human primate genera "
            "after accounting for age, sex, and tooth class."
        )
    else:
        explanation_parts.append(
            "Because the human coefficient is not both positive and statistically significant at the 0.05 level, "
            "the model does not provide strong evidence that modern humans have higher frequencies of antemortem tooth loss than "
            "the non-human primate genera after accounting for age, sex, and tooth class."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON-only output file
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

