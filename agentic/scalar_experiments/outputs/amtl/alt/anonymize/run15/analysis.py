import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns to something more descriptive for modeling
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex_estimate",
            "feature8": "genus",
        }
    )

    # Drop rows with zero sockets or missing key values
    df = df[df["n_sockets"] > 0].copy()
    df = df.dropna(subset=["n_missing", "n_sockets", "age", "sex_estimate", "tooth_class", "genus"])

    # Create response as proportion and use Binomial family with weights
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Set genus reference category to a non-human primate (Pan if present, else Papio/Pongo)
    if "Pan" in df["genus"].unique():
        ref_genus = "Pan"
    elif "Papio" in df["genus"].unique():
        ref_genus = "Papio"
    else:
        ref_genus = sorted(g for g in df["genus"].unique() if g != "Homo sapiens")[0]

    df["genus"] = df["genus"].astype("category")
    df["genus"] = df["genus"].cat.reorder_categories(
        [ref_genus] + [g for g in df["genus"].cat.categories if g != ref_genus],
        ordered=False,
    )

    # Model: prop_missing ~ genus + age + sex_estimate + tooth_class
    formula = "prop_missing ~ C(genus) + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    # Summarize the genus effects, focusing on Homo sapiens vs reference
    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues

    # Collect genus effect stats
    genus_rows = []
    for name in params.index:
        if name.startswith("C(genus)[T.") and "Homo sapiens" in name:
            genus_rows.append(
                {
                    "term": name,
                    "coef": params[name],
                    "ci_lower": conf.loc[name, 0],
                    "ci_upper": conf.loc[name, 1],
                    "pvalue": pvalues[name],
                }
            )

    # Also compute predicted mean AMTL probability for each genus at typical covariate values
    typical = {
        "age": df["age"].median(),
        "sex_estimate": df["sex_estimate"].median(),
        "tooth_class": df["tooth_class"].mode()[0],
    }
    pred_rows = []
    for genus in df["genus"].cat.categories:
        new_df = pd.DataFrame(
            {
                "genus": [genus],
                "age": [typical["age"]],
                "sex_estimate": [typical["sex_estimate"]],
                "tooth_class": [typical["tooth_class"]],
            }
        )
        pred = result.predict(new_df)[0]
        pred_rows.append({"genus": genus, "pred_prop_missing": float(pred)})

    summary = {
        "ref_genus": ref_genus,
        "genus_effects": genus_rows,
        "predictions": pred_rows,
        "model_converged": result.mle_retvals.get("converged", True)
        if hasattr(result, "mle_retvals")
        else True,
        "deviance": float(result.deviance),
        "df_resid": int(result.df_resid),
    }

    # Print a compact JSON-like summary to stdout for the agent to read
    import json

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

