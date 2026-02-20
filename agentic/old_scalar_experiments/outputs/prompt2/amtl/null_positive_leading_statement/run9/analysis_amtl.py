import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and filtering
    df = df[df["sockets"] > 0].copy()
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

    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(target_genera)].copy()

    print("Number of rows after filtering:", len(df))
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    # Raw AMTL rates by genus
    genus_group = df.groupby("genus").agg(
        total_amtl=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
        mean_age=("age", "mean"),
    )
    genus_group["amtl_rate"] = genus_group["total_amtl"] / genus_group["total_sockets"]

    print("\nRaw AMTL rates (num_amtl / sockets) by genus:")
    print(genus_group[["total_amtl", "total_sockets", "amtl_rate", "mean_age"]])

    # Binomial GLM: model proportion of missing teeth with weights = number of sockets
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    formula = "amtl_prop ~ C(genus) + C(tooth_class) + age + prob_male"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nGLM binomial summary:")
    print(result.summary())

    # Predicted probabilities per socket for each genus,
    # holding age, sex, and tooth class at their observed values.
    print(
        "\nPredicted AMTL probabilities per socket by genus "
        "(controlling for age, sex, and tooth class):"
    )
    pred_means: dict[str, float] = {}
    for g in target_genera:
        df_g = df.copy()
        df_g["genus"] = g
        pred = result.predict(df_g)
        pred_means[g] = float(pred.mean())
        print(f"{g}: {pred_means[g]:.4f}")

    homo = pred_means.get("Homo sapiens")
    if homo is not None:
        print("\nDifferences in predicted probability (Homo sapiens - other genus):")
        for g in target_genera:
            if g == "Homo sapiens":
                continue
            diff = homo - pred_means[g]
            print(f"Homo sapiens - {g}: {diff:.4f}")

    # Genus coefficients and p-values from the GLM
    print("\nGenus coefficient estimates and p-values:")
    params = result.params
    pvalues = result.pvalues
    for name in params.index:
        if "genus" in name:
            print(f"{name}: coef={params[name]:.4f}, p={pvalues[name]:.4g}")


if __name__ == "__main__":
    main()

