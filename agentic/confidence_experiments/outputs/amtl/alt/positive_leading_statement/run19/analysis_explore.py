import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy as pt


def main() -> None:
    base_path = Path(__file__).parent

    info_path = base_path / "info.json"
    data_path = base_path / "amtl.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    print("Research question:")
    print(question)
    print()

    df = pd.read_csv(data_path)

    # Basic cleaning: drop rows with missing key fields and require sockets > 0
    df = df.dropna(
        subset=["num_amtl", "sockets", "genus", "age", "prob_male", "tooth_class", "specimen"]
    )
    df = df[df["sockets"] > 0].copy()

    # Ensure categorical types
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")
    df["specimen"] = df["specimen"].astype("category")

    # Set Homo sapiens as the reference level for genus
    if "Homo sapiens" not in list(df["genus"].cat.categories):
        raise ValueError("Expected 'Homo sapiens' genus level not found in data.")

    # Reorder so Homo sapiens is first category (reference)
    other_genera = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
    df["genus"] = df["genus"].cat.reorder_categories(
        ["Homo sapiens"] + other_genera, ordered=False
    )

    print("Genus counts:")
    print(df["genus"].value_counts())
    print()

    print("Summary of sockets and num_amtl:")
    print(df[["num_amtl", "sockets"]].describe())
    print()

    # Sanity checks for binomial GLM
    prop = df["num_amtl"] / df["sockets"]
    print("Proportion num_amtl/sockets outside [0, 1]:")
    print(((prop < 0) | (prop > 1)).sum())
    print("Number of rows with sockets <= 0:")
    print((df["sockets"] <= 0).sum())
    print("Number of rows with sockets == 0:")
    print((df["sockets"] == 0).sum())
    print()

    # Binomial GLM with 2-column response: [successes, failures]
    # Model: AMTL probability ~ genus + age + sex + tooth_class
    exog_formula = (
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    X = pt.dmatrix(exog_formula, df, return_type="dataframe")
    y = np.column_stack(
        [
            df["num_amtl"].to_numpy(),
            (df["sockets"] - df["num_amtl"]).to_numpy(),
        ]
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())

    # Fit model with cluster-robust SEs clustered by specimen
    robust = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    print("Binomial GLM with cluster-robust SEs (cluster = specimen):")
    print(robust.summary())
    print()

    # Extract genus effects: coefficients are non-human genera relative to Homo sapiens
    params = robust.params
    conf_int = robust.conf_int()
    pvalues = robust.pvalues

    print("Genus effects relative to Homo sapiens (log-odds scale and odds ratios):")
    for name in params.index:
        if not name.startswith("C(genus"):
            continue
        coef = params[name]
        ci_low, ci_high = conf_int.loc[name]
        pval = pvalues[name]
        odds_ratio = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
        print(
            f"{name}: coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
            f"p={pval:.3g}, OR={odds_ratio:.3f} (95% CI [{or_low:.3f}, {or_high:.3f}])"
        )
    print()

    # Compute adjusted predicted AMTL probabilities for each genus, averaging over the
    # observed distribution of covariates and weighting by number of sockets.
    unique_genera = list(df["genus"].cat.categories)
    avg_probs = {}
    for genus in unique_genera:
        df_genus = df.copy()
        df_genus["genus"] = genus
        # Build design matrix for this counterfactual dataset
        X_genus = pt.build_design_matrices(
            [X.design_info], df_genus, return_type="dataframe"
        )[0]
        preds = robust.model.predict(robust.params, X_genus)
        avg_prob = float(np.average(preds, weights=df_genus["sockets"]))
        avg_probs[genus] = avg_prob

    print("Adjusted average predicted AMTL probability by genus (socket-weighted):")
    for genus, prob in avg_probs.items():
        print(f"{genus}: {prob:.4f}")

    # Differences relative to Homo sapiens
    homo_prob = avg_probs["Homo sapiens"]
    print()
    print("Differences in adjusted AMTL probability vs Homo sapiens:")
    for genus, prob in avg_probs.items():
        if genus == "Homo sapiens":
            continue
        diff = homo_prob - prob
        print(f"Homo sapiens - {genus}: {diff:.4f}")


if __name__ == "__main__":
    main()
