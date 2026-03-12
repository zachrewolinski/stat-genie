import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: keep rows with valid counts and sockets > 0
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Restrict to target genera
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Construct outcome as proportion with binomial weights
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Use probability of being male as sex proxy; ensure finite
    if df["prob_male"].isna().any():
        df["prob_male"] = df["prob_male"].fillna(df["prob_male"].mean())

    # Genus as categorical with Homo sapiens as reference
    df["genus"] = pd.Categorical(
        df["genus"], categories=["Homo sapiens", "Pan", "Papio", "Pongo"]
    )

    # Fit binomial GLM with logit link
    formula = "amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract genus coefficients (non-human vs human baseline)
    params = result.params
    pvalues = result.pvalues

    # By construction, Homo sapiens is reference; coefficients are
    # log-odds differences for each non-human genus vs humans.
    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus)[T.{genus}]"
        if term in params.index:
            genus_effects[genus] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
            }

    # Compute predicted AMTL probabilities for humans vs non-humans
    # at the mean of covariates.
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    # Use most common tooth_class as representative
    mode_tooth = df["tooth_class"].mode().iloc[0]

    def predict_for_genus(genus_name: str) -> float:
        pred_df = pd.DataFrame(
            {
                "genus": [genus_name],
                "age": [mean_age],
                "prob_male": [mean_prob_male],
                "tooth_class": [mode_tooth],
            }
        )
        return float(result.predict(pred_df)[0])

    human_prob = predict_for_genus("Homo sapiens")
    nonhuman_probs = {
        genus: predict_for_genus(genus) for genus in ["Pan", "Papio", "Pongo"]
    }

    # Summarize descriptive statistics by genus
    genus_summary = (
        df.assign(prop=df["num_amtl"] / df["sockets"])
        .groupby("genus")["prop"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Print a concise JSON summary to stdout so it can be inspected.
    summary = {
        "genus_effects": genus_effects,
        "human_pred_amtl_prob": human_prob,
        "nonhuman_pred_amtl_probs": nonhuman_probs,
        "genus_descriptive_summary": genus_summary.to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

