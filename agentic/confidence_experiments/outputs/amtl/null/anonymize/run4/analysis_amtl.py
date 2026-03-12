import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df = df[df["sockets"] > 0].copy()
    df["prop_missing"] = df["missing"] / df["sockets"]

    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    if "Homo sapiens" not in df["genus"].cat.categories:
        print("Expected 'Homo sapiens' category not found in genus column.")
        print("Observed genera:", df["genus"].unique())
        return

    formula = (
        "prop_missing ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + sex + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Raw AMTL proportions by genus (unadjusted):")
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("missing", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .assign(raw_prop_missing=lambda g: g["total_missing"] / g["total_sockets"])
    )
    print(genus_summary)

    print("\nGLM summary (binomial, logit link):")
    print(result.summary())

    genera = df["genus"].cat.categories.tolist()
    adjusted_probs = {}
    for g in genera:
        new_data = df.copy()
        new_data["genus"] = g
        preds = result.predict(new_data)
        adjusted_probs[g] = float(preds.mean())

    print("\nAdjusted mean predicted AMTL probabilities by genus:")
    for g, p in adjusted_probs.items():
        print(f"  {g}: {p:.4f}")

    nonhuman = [g for g in genera if g != "Homo sapiens"]
    for g in nonhuman:
        param_name = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        coef = float(result.params.get(param_name, np.nan))
        se = float(result.bse.get(param_name, np.nan))
        pval = float(result.pvalues.get(param_name, np.nan))
        oratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        diff = adjusted_probs["Homo sapiens"] - adjusted_probs[g]

        print(f"\nGenus {g} vs Homo sapiens:")
        print(f"  Log-odds difference (genus - Homo): {coef:.4f}")
        print(f"  Std. error: {se:.4f}")
        print(f"  Odds ratio (genus/Homo): {oratio:.3f}")
        print(f"  p-value for difference: {pval:.4g}")
        print(f"  Adjusted probability difference (Homo - {g}): {diff:.4f}")


if __name__ == "__main__":
    main()
