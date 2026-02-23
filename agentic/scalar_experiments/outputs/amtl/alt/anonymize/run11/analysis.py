import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    df["missing"] = df["feature3"]
    df["sockets"] = df["feature4"]
    df["prop_missing"] = df["missing"] / df["sockets"]

    df = df[df["sockets"] > 0].copy()

    df["genus"] = df["feature8"].astype(str)
    df["tooth_class"] = df["feature1"].astype(str)

    # Binomial regression with Homo sapiens as reference genus.
    formula = (
        "prop_missing ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "feature5 + feature7 + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    params = result.params.to_dict()
    pvalues = result.pvalues.to_dict()

    non_human_genera = sorted(
        g for g in df["genus"].unique() if g != "Homo sapiens"
    )

    genus_terms = {
        g: f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        for g in non_human_genera
    }

    genus_summary = {}
    n_sig_human_higher = 0
    n_sig_human_lower = 0

    for genus, term in genus_terms.items():
        coef = float(params.get(term, np.nan))
        pval = float(pvalues.get(term, np.nan))

        # Negative coefficient means that genus has lower AMTL than humans,
        # because Homo sapiens is the reference category.
        if np.isnan(coef) or np.isnan(pval):
            direction = "undetermined"
            significant = False
        else:
            direction = (
                "lower_than_human" if coef < 0 else "higher_than_human"
            )
            significant = bool(pval < 0.05)

        if significant and coef < 0:
            n_sig_human_higher += 1
        if significant and coef > 0:
            n_sig_human_lower += 1

        genus_summary[genus] = {
            "coef": coef,
            "pvalue": pval,
            "direction": direction,
            "significant": significant,
        }

    df["predicted_prop"] = result.predict(df)
    predicted_by_genus = (
        df.groupby("genus")["predicted_prop"].mean().to_dict()
    )

    analysis_results = {
        "genus_terms": genus_summary,
        "predicted_by_genus": predicted_by_genus,
        "n_sig_human_higher": n_sig_human_higher,
        "n_sig_human_lower": n_sig_human_lower,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)


if __name__ == "__main__":
    main()

