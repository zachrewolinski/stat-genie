import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of antemortem tooth loss per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial GLM with Homo sapiens as reference genus, controlling for age, sex, and tooth class.
    formula = (
        "prop_amtl ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract genus coefficients and p-values
    genus_params = {
        name: (coef, result.pvalues[name])
        for name, coef in result.params.items()
        if name.startswith("C(genus")
    }

    # Compute predicted AMTL probabilities for a typical specimen by genus
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_class_mode = df["tooth_class"].mode(dropna=True).iat[0]

    genera = sorted(df["genus"].unique())
    pred_rows = []
    for g in genera:
        pred_rows.append(
            {
                "genus": g,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": tooth_class_mode,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_probs = result.predict(pred_df)

    genus_predictions = {
        g: float(p) for g, p in zip(pred_df["genus"], pred_probs)
    }

    # Summarize key evidence for later interpretation (printed to stdout)
    print("Binomial GLM results (Homo sapiens reference):")
    print(result.summary())
    print("\nGenus coefficients (coef, p-value):")
    for name, (coef, pval) in genus_params.items():
        print(f"{name}: coef={coef:.3f}, p-value={pval:.3g}")

    print("\nPredicted AMTL probability by genus for a typical specimen:")
    for g in genera:
        print(f"{g}: {genus_predictions[g]:.3f}")

    # Simple numeric summary of whether humans have higher AMTL than non-humans
    human_prob = genus_predictions.get("Homo sapiens", np.nan)
    nonhuman_probs = [
        v for k, v in genus_predictions.items() if k != "Homo sapiens"
    ]
    nonhuman_mean = float(np.mean(nonhuman_probs)) if nonhuman_probs else np.nan

    print(
        f"\nTypical predicted AMTL probability - Homo sapiens: {human_prob:.3f}, "
        f"non-human mean: {nonhuman_mean:.3f}"
    )

    # This script does not write conclusion.txt directly; that is constructed
    # after reviewing these results.


if __name__ == "__main__":
    main()

