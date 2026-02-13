import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only the genera relevant to the research question
    valid_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(valid_genera)].copy()

    # Basic sanity filtering
    df = df[
        (df["sockets"] > 0)
        & (df["num_amtl"] >= 0)
        & (df["num_amtl"] <= df["sockets"])
    ].copy()

    # Indicator for humans vs. non-humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    return df


def genus_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            n_specimens=("specimen", "nunique"),
            n_rows=("specimen", "size"),
        )
        .reset_index()
    )
    grouped["prop_amtl"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression of AMTL proportion on human indicator and covariates
    model = smf.glm(
        formula="prop_amtl ~ is_human + C(tooth_class) + age + prob_male",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    res = model.fit()

    # Try to get cluster-robust SEs by specimen; fall back if unavailable
    try:
        res_robust = res.get_robustcov_results(
            cov_type="cluster", groups=df["specimen"]
        )
    except Exception:
        res_robust = res

    return res, res_robust


def summarize_human_effect(res_robust) -> dict:
    params = res_robust.params
    bse = res_robust.bse
    pvalues = res_robust.pvalues

    coef = float(params.get("is_human", np.nan))
    se = float(bse.get("is_human", np.nan))
    pval = float(pvalues.get("is_human", np.nan))

    # Predicted probabilities for a "typical" case
    # Construct representative profiles using mean age/sex and most common tooth class
    mean_age = float(res_robust.model.data.frame["age"].mean())
    mean_prob_male = float(res_robust.model.data.frame["prob_male"].mean())
    mode_tooth_class = (
        res_robust.model.data.frame["tooth_class"].mode().iloc[0]
    )

    design = res_robust.model.data.frame[
        ["age", "prob_male", "tooth_class"]
    ].iloc[[0]].copy()
    design.loc[:, "age"] = mean_age
    design.loc[:, "prob_male"] = mean_prob_male
    design.loc[:, "tooth_class"] = mode_tooth_class

    # Scenario: non-human vs human, holding other covariates constant
    design_nonhuman = design.copy()
    design_nonhuman["is_human"] = 0

    design_human = design.copy()
    design_human["is_human"] = 1

    # Use the robust result's prediction machinery
    pred_nonhuman = float(
        res_robust.predict(design_nonhuman).mean()
    )
    pred_human = float(
        res_robust.predict(design_human).mean()
    )

    return {
        "coef_is_human": coef,
        "se_is_human": se,
        "pvalue_is_human": pval,
        "pred_nonhuman": pred_nonhuman,
        "pred_human": pred_human,
    }


def main():
    df = load_and_prepare_data("amtl.csv")

    genus_summary = genus_level_summary(df)
    res, res_robust = fit_binomial_model(df)
    human_effect = summarize_human_effect(res_robust)

    output = {
        "genus_summary": genus_summary.to_dict(orient="records"),
        "human_effect": human_effect,
        "model_summary": str(res_robust.summary()),
    }

    # Print full analysis details for inspection
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

