import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Inspect genus categories to identify humans vs non-humans
    genus_counts = df["genus"].value_counts()

    # Crude observed AMTL proportions by genus (weighted by sockets)
    grouped = df.groupby("genus", observed=True).agg(
        total_amtl=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    grouped["obs_prop"] = grouped["total_amtl"] / grouped["total_sockets"]

    # Choose the human genus label
    human_labels = [g for g in grouped.index if "homo" in g.lower()]
    human_label = human_labels[0] if human_labels else None

    # Fit binomial GLM on AMTL proportion with sockets as binomial denominator
    # Use human genus as reference if available, otherwise let statsmodels choose.
    if human_label is not None:
        formula = (
            f"prop_amtl ~ C(genus, Treatment(reference='{human_label}'))"
            " + age + prob_male + C(tooth_class)"
        )
    else:
        formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Predicted probabilities for each genus, averaging over observed covariates
    genuses = grouped.index.tolist()
    mean_preds = {}
    for g in genuses:
        df_g = df.copy()
        df_g["genus"] = g
        mean_preds[g] = float(result.predict(df_g).mean())

    # Extract coefficient tests comparing each non-human genus to humans
    genus_params = {
        name: (float(coef), float(pval))
        for name, coef, pval in zip(
            result.params.index, result.params.values, result.pvalues.values
        )
        if name.startswith("C(genus")
    }

    # Summarize key results in a JSON blob we can inspect from the shell.
    output = {
        "genus_counts": genus_counts.to_dict(),
        "observed_by_genus": grouped["obs_prop"].to_dict(),
        "mean_pred_by_genus": mean_preds,
        "genus_params": genus_params,
        "human_label": human_label,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
