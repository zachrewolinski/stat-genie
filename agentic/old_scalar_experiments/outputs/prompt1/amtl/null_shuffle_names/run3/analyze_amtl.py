import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv("amtl.csv")

    # Map column names to clearer semantics based on info.json description.
    df = df.rename(
        columns={
            "genus": "num_missing",  # number of missing teeth for this record
            "age": "num_sockets",  # number of observable sockets
            "pop": "age_estimate",  # estimated age at death (years)
            "stdev_age": "sex_estimate",  # proxy for sex (0=female,1=male, in 0.25 steps)
            "tooth_class": "species",  # Homo sapiens, Pan, Papio, Pongo
            "sockets": "tooth_region",  # Anterior / Posterior / Premolar
        }
    )

    # Basic cleaning: keep rows with sensible counts.
    df = df[df["num_sockets"] > 0].copy()
    df = df[df["num_missing"] >= 0].copy()

    # Drop clearly inconsistent rows where missing teeth exceed sockets.
    df = df[df["num_missing"] <= df["num_sockets"]].copy()

    # Compute per-row missing fraction.
    df["missing_frac"] = df["num_missing"] / df["num_sockets"]

    # Ensure categorical ordering with Homo sapiens as baseline.
    df["species"] = pd.Categorical(
        df["species"], categories=["Homo sapiens", "Pan", "Papio", "Pongo"]
    )
    df["tooth_region"] = df["tooth_region"].astype("category")

    return df


def summarize_by_genus(df: pd.DataFrame) -> pd.DataFrame:
    # Weighted mean missing fraction by genus, using sockets as weights.
    def wmean(x, w):
        return np.average(x, weights=w)

    grouped = (
        df.groupby("species")
        .apply(lambda g: pd.Series(
            {
                "weighted_missing_frac": wmean(g["missing_frac"], g["num_sockets"]),
                "mean_age": g["age_estimate"].mean(),
                "mean_sex_estimate": g["sex_estimate"].mean(),
                "total_sockets": g["num_sockets"].sum(),
                "total_missing": g["num_missing"].sum(),
            }
        ))
        .reset_index()
    )
    return grouped


def fit_binomial_model(df: pd.DataFrame):
    # Binomial GLM on proportions with counts as frequency weights.
    formula = "missing_frac ~ C(species) + age_estimate + sex_estimate + C(tooth_region)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def main() -> None:
    df = load_and_prepare()

    # Descriptive summary by genus.
    genus_summary = summarize_by_genus(df)
    print("Weighted missing fraction by genus/species:")
    print(genus_summary)

    # Fit regression model.
    result = fit_binomial_model(df)
    print("\nBinomial regression summary:")
    print(result.summary())

    # Extract coefficients for non-human genera relative to Homo sapiens baseline.
    params = result.params
    conf_int = result.conf_int()

    genus_effects = {}
    for other in ["Pan", "Papio", "Pongo"]:
        term = f"C(species)[T.{other}]"
        if term in params.index:
            genus_effects[other] = {
                "coef": params[term],
                "lower": conf_int.loc[term, 0],
                "upper": conf_int.loc[term, 1],
                "pvalue": result.pvalues[term],
            }

    print("\nGenus effects relative to Homo sapiens:")
    for g, stats in genus_effects.items():
        print(
            f"{g}: coef={stats['coef']:.3f}, 95% CI=({stats['lower']:.3f}, {stats['upper']:.3f}), "
            f"p={stats['pvalue']:.3g}"
        )

    # Decide answer based on whether all non-human genera have significantly lower log-odds
    # of AMTL than Homo sapiens (coef < 0 and CI entirely below 0).
    all_lower = True
    for stats in genus_effects.values():
        if not (stats["upper"] < 0 and stats["coef"] < 0):
            all_lower = False
            break

    response = "Yes" if all_lower else "No"

    # Prepare a concise machine-readable summary to help build conclusion.txt later if desired.
    summary = {
        "response": response,
        "genus_effects": genus_effects,
        "weighted_missing_by_genus": genus_summary.to_dict(orient="records"),
    }

    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAnalysis decision:", response)
    print("Details written to analysis_summary.json")


if __name__ == "__main__":
    main()

