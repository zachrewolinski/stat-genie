import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    df = df.copy()
    # Proportion of antemortem tooth loss
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    # Binary indicator for humans vs non-human primates
    df["is_human"] = (df["genus"].str.contains("Homo", case=False)).astype(int)
    return df


def fit_binary_human_model(df: pd.DataFrame):
    """
    Binomial GLM with logit link:
    response: proportion AMTL with sockets as binomial denominator
    predictors: is_human + age + prob_male + tooth_class
    """
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def fit_genus_model(df: pd.DataFrame):
    """
    Binomial GLM with full genus factor to examine pairwise differences.
    Use Homo sapiens as the reference category if present.
    """
    # Ensure genus is treated as categorical with an explicit order
    df = df.copy()
    if "Homo sapiens" in df["genus"].unique():
        df["genus"] = pd.Categorical(
            df["genus"], categories=["Homo sapiens", "Pan", "Papio", "Pongo"]
        )
    model = smf.glm(
        formula="amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_means(df: pd.DataFrame):
    genus_stats = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": len(g),
                    "mean_prop": np.average(g["amtl_prop"], weights=g["sockets"]),
                }
            )
        )
        .reset_index()
    )
    return genus_stats


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    print("Unique genera:", df["genus"].unique())
    print("\nWeighted mean AMTL proportion by genus:")
    print(summarize_genus_means(df))

    print("\n=== Binomial GLM: Human vs non-human (is_human) ===")
    human_model = fit_binary_human_model(df)
    print(human_model.summary())

    print("\n=== Binomial GLM: Full genus factor (Homo sapiens reference if available) ===")
    genus_model = fit_genus_model(df)
    print(genus_model.summary())

    # Save a small JSON with key statistics that can be inspected if needed
    # (not the final conclusion.txt required by the task).
    coef = human_model.params.get("is_human", np.nan)
    pval = human_model.pvalues.get("is_human", np.nan)
    out = {
        "human_coef_logit": float(coef) if np.isfinite(coef) else None,
        "human_p_value": float(pval) if np.isfinite(pval) else None,
    }
    Path("model_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

