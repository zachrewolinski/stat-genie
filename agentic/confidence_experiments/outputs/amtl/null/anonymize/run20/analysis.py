import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Proportion of missing teeth
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Ensure genus is categorical with Homo sapiens as reference
    df["genus"] = pd.Categorical(df["genus"])
    if "Homo sapiens" in df["genus"].cat.categories:
        other = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
        df["genus"] = df["genus"].cat.reorder_categories(
            ["Homo sapiens", *other], ordered=False
        )

    # Binomial regression: logit(prop_missing) with n_sockets as weights
    formula = "prop_missing ~ C(genus) + C(tooth_class) + age + sex_estimate"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    print("=== Binomial GLM summary ===")
    print(result.summary())

    # Coefficients and odds ratios for genera (relative to Homo sapiens)
    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    genus_rows = {k: v for k, v in params.items() if k.startswith("C(genus)[T.")}
    print("\n=== Genus effects (relative to Homo sapiens) ===")
    genus_effects = []
    for name, coef in genus_rows.items():
        genus = name.split("[T.")[-1].rstrip("]")
        or_value = float(np.exp(coef))
        pval = float(pvalues[name])
        se = float(bse[name])
        genus_effects.append(
            {
                "genus": genus,
                "coef": float(coef),
                "se": se,
                "odds_ratio": or_value,
                "p_value": pval,
            }
        )
        print(
            f"{genus:8s}: coef={coef: .3f}, SE={se: .3f}, "
            f"OR={or_value: .3f}, p={pval: .4f}"
        )

    # Predicted AMTL frequency at reference covariates for each genus
    mean_age = float(df["age"].mean())
    mean_sex = float(df["sex_estimate"].mean())
    ref_tooth = df["tooth_class"].mode().iat[0]

    print("\n=== Predicted AMTL proportion by genus ===")
    pred_rows = []
    for genus in df["genus"].cat.categories:
        pred_df = pd.DataFrame(
            {
                "genus": [genus],
                "tooth_class": [ref_tooth],
                "age": [mean_age],
                "sex_estimate": [mean_sex],
                "n_sockets": [1.0],
            }
        )
        pred_prob = float(result.predict(pred_df)[0])
        pred_rows.append({"genus": genus, "pred_prop_missing": pred_prob})
        print(f"{genus:12s}: predicted prop_missing = {pred_prob: .3f}")

    # Save a compact JSON summary for inspection if needed
    summary = {
        "genus_effects": genus_effects,
        "predicted_props": pred_rows,
        "mean_age": mean_age,
        "mean_sex_estimate": mean_sex,
        "reference_tooth_class": ref_tooth,
    }
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

