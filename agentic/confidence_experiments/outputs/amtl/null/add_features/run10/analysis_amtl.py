import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep rows with valid numbers of sockets
    df = df[df["sockets"] > 0].copy()

    # Proportion of antemortem tooth loss per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Treat key predictors as categorical
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Ensure "Homo sapiens" is included as a category
    if "Homo sapiens" not in list(df["genus"].cat.categories):
        raise ValueError("Expected 'Homo sapiens' to be present in genus categories.")

    # Binomial regression with Homo sapiens as reference genus
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

    print(result.summary())

    # Adjusted mean predicted AMTL proportions by genus:
    genus_levels = df["genus"].cat.categories
    mean_preds: dict[str, float] = {}

    for g in genus_levels:
        df_g = df.copy()
        df_g["genus"] = pd.Categorical(
            [g] * len(df_g),
            categories=genus_levels,
        )
        mean_preds[str(g)] = float(result.predict(df_g).mean())

    print("\nAdjusted mean predicted AMTL proportions by genus (model-based):")
    for g, val in mean_preds.items():
        print(f"{g}: {val:.4f}")

    # Also show raw (unadjusted) mean proportions by genus for context
    raw_means = (
        df.assign(prop_amtl=df["num_amtl"] / df["sockets"])
        .groupby("genus")["prop_amtl"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nRaw mean AMTL proportions by genus (unadjusted):")
    for g, val in raw_means.items():
        print(f"{g}: {val:.4f}")

    # Highlight genus-related coefficients and p-values
    print("\nGenus-related coefficients (relative to Homo sapiens):")
    for name in result.params.index:
        if "genus" in name:
            coef = float(result.params[name])
            pval = float(result.pvalues[name])
            print(f"{name}: coef={coef:.4f}, p={pval:.4g}")


if __name__ == "__main__":
    main()

