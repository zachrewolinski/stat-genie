import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    df = df.copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Ensure genera are as expected
    print("Genus value counts:")
    print(df["genus"].value_counts())
    print()

    # Binomial regression: proportion of missing teeth with sockets as trials
    # Use Homo sapiens as the reference category for genus (default alphabetical).
    formula = "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print(model.summary())

    # Extract coefficient and p-values for genus terms
    print("\nGenus coefficients (relative to Homo sapiens):")
    for name, coef in model.params.items():
        if name.startswith("C(genus)"):
            pval = model.pvalues[name]
            print(f"{name}: coef={coef:.3f}, p={pval:.4g}")

    # Standardised predicted AMTL rates by genus
    genera = sorted(df["genus"].unique())
    mean_predictions = {}

    # Standardize over the empirical joint distribution of age, sex, and tooth_class
    base = df.copy()
    for g in genera:
        g_df = base.copy()
        g_df["genus"] = g
        preds = model.predict(g_df)
        mean_predictions[g] = preds.mean()

    print("\nStandardised mean predicted AMTL rates by genus:")
    for g, p in mean_predictions.items():
        print(f"{g}: {p:.3f}")

    # Direct comparison: humans vs. non-humans
    print("\n--- Human vs. non-human model ---")
    df_h = df.copy()
    df_h["is_human"] = (df_h["genus"] == "Homo sapiens").astype(int)

    formula_h = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model_h = smf.glm(
        formula=formula_h,
        data=df_h,
        family=sm.families.Binomial(),
        freq_weights=df_h["sockets"],
    ).fit()

    print(model_h.summary().tables[1])

    coef = model_h.params["is_human"]
    pval = model_h.pvalues["is_human"]
    print(f"\nHuman effect (is_human coefficient): {coef:.3f}, p={pval:.4g}")

    # Standardised predictions for humans vs. non-humans
    base_h = df_h.copy()
    base_h["is_human"] = 1
    pred_human = model_h.predict(base_h).mean()
    base_h["is_human"] = 0
    pred_nonhuman = model_h.predict(base_h).mean()
    print(
        f"Standardised mean predicted AMTL rate - humans: {pred_human:.3f}, "
        f"non-humans: {pred_nonhuman:.3f}"
    )


if __name__ == "__main__":
    main()
