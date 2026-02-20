import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "missing",
            "feature4": "observable",
            "feature5": "age",
            "feature7": "sex",
            "feature8": "genus",
        }
    )

    df = df[df["observable"] > 0].copy()

    genera_keep = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_keep)].copy()

    df["prop_missing"] = df["missing"] / df["observable"]

    print("Rows after filtering:", len(df))
    print("AMTL proportion by genus (raw):")
    print(df.groupby("genus")["prop_missing"].mean())

    formula = "prop_missing ~ C(genus) + C(tooth_class) + age + sex"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable"],
    )
    result = model.fit()

    print(result.summary())

    age_val = df["age"].mean()
    sex_val = df["sex"].mean()
    tooth_classes = df["tooth_class"].unique()

    rows = []
    for genus in genera_keep:
        for tooth_class in tooth_classes:
            rows.append(
                {
                    "genus": genus,
                    "tooth_class": tooth_class,
                    "age": age_val,
                    "sex": sex_val,
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_probs = result.predict(pred_df)
    pred_df["pred_prob"] = pred_probs

    print("\nPredicted AMTL probability by genus (averaged over tooth class):")
    print(pred_df.groupby("genus")["pred_prob"].mean())

    print("\nGenus coefficients (vs Homo sapiens baseline):")
    for genus in genera_keep:
        if genus == "Homo sapiens":
            continue
        term = f"C(genus)[T.{genus}]"
        if term in result.params.index:
            coef = result.params[term]
            pval = result.pvalues[term]
            print(f"{term}: coef={coef:.3f}, p={pval:.4g}")
        else:
            print(f"{term} not in model terms")


if __name__ == "__main__":
    main()

