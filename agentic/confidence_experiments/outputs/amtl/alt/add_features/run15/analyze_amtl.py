import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base = Path(__file__).parent
    df = pd.read_csv(base / "amtl.csv")

    # Keep variables relevant to the research question.
    df = df[
        [
            "tooth_class",
            "specimen",
            "num_amtl",
            "sockets",
            "age",
            "stdev_age",
            "prob_male",
            "genus",
        ]
    ].copy()

    # Drop any rows with missing key fields or zero sockets (cannot define a proportion).
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["sockets"] > 0]

    # Restrict to genera of interest.
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Create binary indicator for Homo sapiens vs. non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Center/scale continuous predictors for numerical stability.
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Descriptive AMTL proportion by genus.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    genus_summary = (
        df.groupby("genus")["prop_amtl"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_prop", "std": "sd_prop", "count": "n"})
    )
    genus_summary_dict = genus_summary.reset_index().to_dict(orient="records")

    # Binomial GLM with logit link for AMTL counts out of sockets.
    # Model: num_amtl ~ is_human + age + sex + tooth_class
    # Use weights equal to number of sockets to avoid some extreme leverage.
    try:
        formula_binom = "prop_amtl ~ is_human + age_c + prob_male_c + C(tooth_class)"
        model_binom = smf.glm(
            formula=formula_binom,
            data=df,
            family=sm.families.Binomial(),
            var_weights=df["sockets"],
        )
        result_binom = model_binom.fit()
        coef = result_binom.params.get("is_human", np.nan)
        se = result_binom.bse.get("is_human", np.nan)
        pval = result_binom.pvalues.get("is_human", np.nan)
    except Exception:
        # Fall back to linear model on proportions if binomial GLM fails.
        formula_lm = "prop_amtl ~ is_human + age_c + prob_male_c + C(tooth_class)"
        model_binom = smf.wls(
            formula=formula_lm,
            data=df,
            weights=df["sockets"],
        )
        result_binom = model_binom.fit()
        coef = result_binom.params.get("is_human", np.nan)
        se = result_binom.bse.get("is_human", np.nan)
        pval = result_binom.pvalues.get("is_human", np.nan)

    # Compute adjusted mean AMTL proportion for humans vs non-humans
    # using the regression model at average covariates and the most common tooth class.
    ref_tooth = df["tooth_class"].mode().iat[0]
    mean_age_c = 0.0
    mean_prob_male_c = 0.0

    def predict_prop(is_human: int) -> float:
        new_df = pd.DataFrame(
            {
                "is_human": [is_human],
                "age_c": [mean_age_c],
                "prob_male_c": [mean_prob_male_c],
                "tooth_class": [ref_tooth],
            }
        )
        new_df["tooth_class"] = new_df["tooth_class"].astype(df["tooth_class"].dtype)
        pred = result_binom.predict(new_df)
        return float(pred.iloc[0])

    prop_human = predict_prop(1)
    prop_nonhuman = predict_prop(0)
    diff = prop_human - prop_nonhuman

    summary = {
        "genus_descriptive": genus_summary_dict,
        "coef_is_human": float(coef) if np.isfinite(coef) else None,
        "se_is_human": float(se) if np.isfinite(se) else None,
        "pval_is_human": float(pval) if np.isfinite(pval) else None,
        "prop_human_adj": prop_human,
        "prop_nonhuman_adj": prop_nonhuman,
        "diff_prop_adj": diff,
        "n_rows": int(len(df)),
        "n_specimens": int(df["specimen"].nunique()),
        "ref_tooth_class": ref_tooth,
    }

    (base / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
