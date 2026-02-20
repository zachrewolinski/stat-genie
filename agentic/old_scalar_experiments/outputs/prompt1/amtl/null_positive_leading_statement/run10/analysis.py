import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic derived measure: proportion of missing teeth for the tooth class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for humans vs all non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categorical types and set Homo sapiens as reference for genus
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in list(df["genus"].cat.categories):
        new_order = ["Homo sapiens"] + [
            g for g in df["genus"].cat.categories if g != "Homo sapiens"
        ]
        df["genus"] = df["genus"].cat.reorder_categories(new_order, ordered=False)

    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial regression with logit link (genus-specific effects):
    # Response is the proportion missing, with the number of sockets as weights.
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Model 1: Genus-specific effects (Homo sapiens as reference)\n")
    print(result.summary())

    # Genus effects relative to Homo sapiens
    genus_params = {
        name: (param, result.pvalues[name])
        for name, param in result.params.items()
        if name.startswith("C(genus)[T.")
    }

    print("\nGenus coefficients (relative to Homo sapiens):")
    for name, (coef, pval) in genus_params.items():
        print(f"{name}: coef={coef:.4f}, p-value={pval:.4g}")

    # Descriptive: raw mean AMTL proportion by genus
    df["raw_prop"] = df["num_amtl"] / df["sockets"]
    print("\nRaw mean proportion of missing teeth by genus:")
    print(df.groupby("genus")["raw_prop"].mean())

    # Second model: humans vs all non-human primates
    formula2 = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model2 = smf.glm(
        formula=formula2,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result2 = model2.fit()

    print("\n\nModel 2: Humans vs non-human primates\n")
    print(result2.summary())
    print(
        f"\nHuman indicator coefficient: {result2.params['is_human']:.4f}, "
        f"p-value={result2.pvalues['is_human']:.4g}"
    )


if __name__ == "__main__":
    main()
