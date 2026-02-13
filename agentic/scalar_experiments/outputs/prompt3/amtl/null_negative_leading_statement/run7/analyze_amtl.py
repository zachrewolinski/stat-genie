import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a logistic regression of tooth-level AMTL on genus, age, sex, and tooth class.

    The original data are counts (num_amtl out of sockets). We expand these to a
    tooth-level data set with one row per tooth and a binary outcome indicating
    whether that tooth was lost antemortem.
    """
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] <= df["sockets"]]
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "genus",
            "tooth_class",
        ]
    )

    long_rows = []
    for _, row in df.iterrows():
        n_missing = int(row["num_amtl"])
        n_present = int(row["sockets"] - row["num_amtl"])
        base = {
            "genus": row["genus"],
            "age": row["age"],
            "prob_male": row["prob_male"],
            "tooth_class": row["tooth_class"],
        }
        long_rows.extend({**base, "amtl": 1} for _ in range(n_missing))
        long_rows.extend({**base, "amtl": 0} for _ in range(n_present))

    df_long = pd.DataFrame(long_rows)

    formula = "amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.logit(formula=formula, data=df_long)
    result = model.fit(disp=False)
    return result, df_long


def marginal_genus_probabilities(result, df: pd.DataFrame) -> dict:
    """
    Compute marginal predicted AMTL probabilities for each genus.

    For each genus g, we:
      * set the genus column in a copy of the tooth-level data frame to g
      * predict AMTL probabilities for each row
      * take a simple mean so that each tooth contributes equally
    """
    genera = sorted(df["genus"].unique())
    probs = {}
    for g in genera:
        tmp = df.copy()
        tmp["genus"] = g
        pred = result.predict(tmp)
        probs[g] = float(pred.mean())
    return probs


def summarize_genus_effects(result) -> dict:
    """
    Extract coefficients and p-values for genus effects relative to the reference genus.
    """
    params = result.params
    pvalues = result.pvalues
    genus_effects = {}

    for name, coef in params.items():
        if name.startswith("C(genus)"):
            genus_effects[name] = {
                "coef": float(coef),
                "pvalue": float(pvalues[name]),
            }
    return genus_effects


def main():
    data_path = Path("amtl.csv")
    df = load_data(data_path)

    result, df_model = fit_binomial_model(df)
    genus_probs = marginal_genus_probabilities(result, df_model)
    genus_effects = summarize_genus_effects(result)

    # Save a compact JSON summary that can be inspected separately.
    summary = {
        "genus_marginal_probabilities": genus_probs,
        "genus_effects_vs_reference": genus_effects,
        "reference_note": (
            "Genus effects are parameterized with a categorical reference level "
            "used by statsmodels (typically the first genus in alphabetical order). "
            "Negative coefficients indicate lower log-odds of AMTL than the reference; "
            "positive coefficients indicate higher log-odds."
        ),
    }

    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also print a human-readable overview.
    print("=== Marginal predicted AMTL probability by genus ===")
    for g, p in genus_probs.items():
        print(f"{g:15s}: {p:.4f}")

    print("\n=== Genus coefficients (relative to reference) ===")
    for name, info in genus_effects.items():
        print(f"{name:25s} coef={info['coef']:.4f}  p={info['pvalue']:.4g}")


if __name__ == "__main__":
    main()
