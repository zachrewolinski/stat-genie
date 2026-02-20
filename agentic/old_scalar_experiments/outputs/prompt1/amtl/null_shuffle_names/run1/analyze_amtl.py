import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load AMTL dataset and harmonize column semantics."""
    df = pd.read_csv(csv_path)

    # Rename columns to match their semantic meaning from info.json
    df = df.rename(
        columns={
            "sockets": "tooth_class",
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age_at_death",
            "num_amtl": "age_sd",
            "stdev_age": "prob_male_est",
            "tooth_class": "genus",
            "specimen": "region",
        }
    )

    # Basic cleaning
    df = df.dropna(
        subset=["num_missing", "num_sockets", "age_at_death", "prob_male_est", "genus", "tooth_class"]
    ).copy()

    # Ensure counts are integers and valid
    df["num_missing"] = df["num_missing"].astype(int)
    df["num_sockets"] = df["num_sockets"].astype(int)
    df = df[df["num_sockets"] > 0]
    df = df[df["num_missing"].between(0, df["num_sockets"])]

    # Standardize genus labels
    df["genus"] = df["genus"].replace(
        {
            "Homo sapiens": "Homo",
            "Homo Sapiens": "Homo",
        }
    )

    # Focus on the genera relevant to the research question
    genera_of_interest = {"Homo", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Treat factors as categorical
    df["genus"] = pd.Categorical(df["genus"], categories=["Homo", "Pan", "Pongo", "Papio"])
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression model of AMTL frequency."""
    # Use aggregated binomial with proportion and socket counts as weights
    df = df.copy()
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    formula = "prop_missing ~ C(genus, Treatment(reference='Homo')) + age_at_death + prob_male_est + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    ).fit()

    return model


def evaluate_human_vs_others(model) -> dict:
    """Evaluate whether humans have higher AMTL than each non-human genus."""
    summary = {}

    # Coefficients for genera are differences vs Homo (reference)
    params = model.params
    pvalues = model.pvalues

    for genus in ["Pan", "Pongo", "Papio"]:
        coef_name = f"C(genus, Treatment(reference='Homo'))[T.{genus}]"
        if coef_name in params:
            coef = params[coef_name]
            pval = pvalues[coef_name]
            # Negative coefficient => lower AMTL than Homo (since reference is Homo)
            summary[genus] = {"coef": float(coef), "pvalue": float(pval)}

    return summary


def main():
    df = load_and_prepare_data("amtl.csv")
    model = fit_binomial_model(df)

    genus_results = evaluate_human_vs_others(model)

    # Simple decision rule:
    # If all non-human genera have negative coefficients vs Homo and at least one is statistically
    # significantly lower (p < 0.05), we conclude humans have higher AMTL frequencies.
    all_negative = all(info["coef"] < 0 for info in genus_results.values())
    any_significant = any((info["coef"] < 0) and (info["pvalue"] < 0.05) for info in genus_results.values())

    conclusion = {
        "all_negative_vs_humans": all_negative,
        "any_significantly_lower_than_humans": any_significant,
        "genus_effects": genus_results,
        "model_aic": float(model.aic),
    }

    # Persist intermediate results to help manual interpretation
    Path("analysis_results.json").write_text(json.dumps(conclusion, indent=2))

    # Also print to stdout for interactive inspection
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
