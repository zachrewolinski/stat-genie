import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"

    with info_path.open() as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_score",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: keep rows with valid counts and non-missing key variables
    df = df[
        (df["sockets"] > 0)
        & (df["missing"] >= 0)
        & df["age"].notna()
        & df["sex_score"].notna()
        & df["tooth_class"].notna()
        & df["genus"].notna()
    ].copy()

    # For binomial modeling, drop rows with impossible counts (missing > sockets)
    df = df[df["missing"] <= df["sockets"]].copy()

    df["prop_missing"] = df["missing"] / df["sockets"]

    # Fit a binomial GLM with Homo sapiens as the implicit reference genus
    model = smf.glm(
        "prop_missing ~ C(genus) + age + sex_score + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Compute adjusted mean AMTL probability for each genus by
    # predicting for all observations while varying genus
    genera = sorted(df["genus"].unique())
    mean_pred_by_genus: dict[str, float] = {}
    for g in genera:
        tmp = df.copy()
        tmp["genus"] = g
        preds = result.predict(tmp)
        mean_pred_by_genus[g] = float(np.mean(preds))

    # Extract genus coefficients (relative to Homo sapiens baseline)
    params = result.params
    conf_int = result.conf_int()

    genus_effects = {}
    for g in genera:
        if g == "Homo sapiens":
            genus_effects[g] = {
                "coef": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }
        else:
            label = f"C(genus)[T.{g}]"
            if label in params.index:
                coef = float(params[label])
                ci_low, ci_high = conf_int.loc[label]
                genus_effects[g] = {
                    "coef": coef,
                    "ci_lower": float(ci_low),
                    "ci_upper": float(ci_high),
                }

    summary = {
        "question": question,
        "n_rows_used": int(len(df)),
        "genera": genera,
        "mean_pred_by_genus": mean_pred_by_genus,
        "genus_effects_log_odds_vs_human": genus_effects,
    }

    # Print JSON summary to stdout for inspection
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

