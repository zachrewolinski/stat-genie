import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only variables relevant to the research question
    cols = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "genus",
    ]
    df = df[cols].copy()

    # Drop rows with missing values in core variables
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Remove clearly invalid rows where num_amtl exceeds sockets
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Compute proportion of missing teeth
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical variables are treated as such
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_glm(df: pd.DataFrame):
    # Use Homo sapiens as the reference genus so other coefficients are differences vs humans
    formula = "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(result, df: pd.DataFrame):
    # Build a small summary of coefficients and odds ratios for genera vs Homo sapiens
    params = result.params
    conf_int = result.conf_int()

    genus_levels = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]

    summary = {}
    for g in genus_levels:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if term in params.index:
            coef = params[term]
            ci_low, ci_high = conf_int.loc[term]
            # Convert to odds ratio scale
            or_val = float(np.exp(coef))
            or_low = float(np.exp(ci_low))
            or_high = float(np.exp(ci_high))
            pval = float(result.pvalues[term])
            summary[g] = {
                "coef": float(coef),
                "odds_ratio": or_val,
                "ci_low": or_low,
                "ci_high": or_high,
                "p_value": pval,
            }
    return summary


def main():
    base = Path(__file__).parent
    info = load_metadata(base / "info.json")

    df_raw = load_data(base / "amtl.csv")
    print(f"Loaded data with shape: {df_raw.shape}")

    df = prepare_data(df_raw)
    print(f"Data after cleaning has shape: {df.shape}")

    # Print how many rows were dropped due to invalid num_amtl > sockets
    invalid_mask = (df_raw["sockets"] > 0) & (df_raw["num_amtl"] > df_raw["sockets"])
    num_invalid = int(invalid_mask.sum())
    print(f"Number of rows with num_amtl > sockets (dropped): {num_invalid}")

    result = fit_binomial_glm(df)
    print(result.summary())

    genus_summary = summarize_genus_effects(result, df)
    print("\nGenus effects vs Homo sapiens (odds ratios < 1 imply lower AMTL than humans):")
    for g, stats in genus_summary.items():
        print(
            f"{g}: OR={stats['odds_ratio']:.3f}, "
            f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}), "
            f"p={stats['p_value']:.3g}"
        )

    # Also fit a simpler model with a binary human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    formula_binary = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model_binary = smf.glm(
        formula=formula_binary,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result_binary = model_binary.fit()

    print("\nModel with binary human vs non-human indicator:")
    print(result_binary.summary())

    # Summarize the human indicator effect
    coef_human = result_binary.params["is_human"]
    ci_low_h, ci_high_h = result_binary.conf_int().loc["is_human"]
    or_h = float(np.exp(coef_human))
    or_low_h = float(np.exp(ci_low_h))
    or_high_h = float(np.exp(ci_high_h))
    p_h = float(result_binary.pvalues["is_human"])
    print(
        f"\nHuman indicator effect (Homo sapiens vs non-human primates): "
        f"OR={or_h:.3f}, 95% CI=({or_low_h:.3f}, {or_high_h:.3f}), p={p_h:.3g}"
    )


if __name__ == "__main__":
    main()
