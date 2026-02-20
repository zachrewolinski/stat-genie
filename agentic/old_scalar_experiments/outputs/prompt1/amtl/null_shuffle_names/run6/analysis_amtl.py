import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r") as f:
        return json.load(f)


def prepare_data(csv_path: Path, info: dict) -> pd.DataFrame:
    """
    Load the AMTL dataset and reconstruct semantically meaningful variables
    based on the shuffled column names described in info.json.
    """
    df = pd.read_csv(csv_path)

    # Reconstruct variables using descriptions from info.json:
    # sockets        -> tooth class (Anterior/Posterior/Premolar)
    # prob_male      -> specimen identifier
    # genus          -> number of teeth missing in this tooth class
    # age            -> number of observable sockets for this tooth class
    # pop            -> estimated age at death
    # num_amtl       -> uncertainty (SD) of age at death
    # stdev_age      -> probability specimen is male (0–1)
    # tooth_class    -> genus (Homo sapiens, Pan, Pongo, Papio)
    # specimen       -> population/region label

    df = df.copy()
    df["tooth_class_cat"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_est"] = df["pop"].astype(float)
    df["age_sd"] = df["num_amtl"].astype(float)
    df["prob_male_est"] = df["stdev_age"].astype(float)
    df["genus_cat"] = df["tooth_class"]
    df["region"] = df["specimen"]

    # Basic sanity filters
    df = df[df["num_sockets"] > 0].copy()

    # Proportion of missing teeth within tooth class for each specimen row
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Indicator for modern human vs non-human primates
    df["is_human"] = (df["genus_cat"] == "Homo sapiens").astype(int)

    # Center/scale some continuous predictors for numeric stability
    df["age_est_c"] = df["age_est"] - df["age_est"].mean()
    df["prob_male_c"] = df["prob_male_est"] - df["prob_male_est"].mean()

    return df


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL proportion, using grouped
    binomial data with the number of observable sockets as frequency weights.
    """
    formula = "prop_missing ~ is_human + age_est_c + prob_male_c + C(tooth_class_cat)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )

    # Use cluster-robust standard errors at the specimen level to account
    # for multiple rows per specimen.
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen_id"]})
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    """
    Summarize the effect of being human vs non-human on AMTL frequency,
    based on the fitted model and observed data.
    """
    # Extract coefficient and p-value for the human indicator
    coef_human = result.params.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)

    # Predicted marginal probabilities for humans vs non-humans,
    # holding the empirical distribution of covariates fixed.
    df_pred_human = df.copy()
    df_pred_human["is_human"] = 1
    pred_human = result.predict(df_pred_human)

    df_pred_nonhuman = df.copy()
    df_pred_nonhuman["is_human"] = 0
    pred_nonhuman = result.predict(df_pred_nonhuman)

    mean_pred_human = float(pred_human.mean())
    mean_pred_nonhuman = float(pred_nonhuman.mean())

    # Also compute simple observed proportions for context
    # (number of missing teeth / number of sockets) pooled by genus group.
    human_mask = df["is_human"] == 1
    total_missing_human = float(df.loc[human_mask, "num_missing"].sum())
    total_sockets_human = float(df.loc[human_mask, "num_sockets"].sum())

    total_missing_nonhuman = float(df.loc[~human_mask, "num_missing"].sum())
    total_sockets_nonhuman = float(df.loc[~human_mask, "num_sockets"].sum())

    obs_rate_human = total_missing_human / total_sockets_human
    obs_rate_nonhuman = total_missing_nonhuman / total_sockets_nonhuman

    return {
        "coef_human": float(coef_human),
        "p_human": float(p_human),
        "mean_pred_human": mean_pred_human,
        "mean_pred_nonhuman": mean_pred_nonhuman,
        "obs_rate_human": float(obs_rate_human),
        "obs_rate_nonhuman": float(obs_rate_nonhuman),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    info = load_metadata(base_dir / "info.json")
    df = prepare_data(base_dir / "amtl.csv", info)
    result = fit_model(df)
    summary = summarize_effect(df, result)

    # Decide on "Yes" / "No" answer based on direction and significance
    coef = summary["coef_human"]
    pval = summary["p_human"]
    mean_pred_h = summary["mean_pred_human"]
    mean_pred_nh = summary["mean_pred_nonhuman"]

    # Humans have higher AMTL frequency if:
    # - the human coefficient is positive,
    # - predicted AMTL probability for humans exceeds non-humans,
    # - and the effect is statistically significant at alpha=0.05.
    has_higher_amtl = (
        np.isfinite(coef)
        and coef > 0
        and mean_pred_h > mean_pred_nh
        and np.isfinite(pval)
        and pval < 0.05
    )

    conclusion = {
        "has_higher_amtl": has_higher_amtl,
        "coef_human": coef,
        "p_human": pval,
        "mean_pred_human": mean_pred_h,
        "mean_pred_nonhuman": mean_pred_nh,
        "obs_rate_human": summary["obs_rate_human"],
        "obs_rate_nonhuman": summary["obs_rate_nonhuman"],
    }

    # Store an intermediate summary to aid interpretation if needed.
    (base_dir / "analysis_summary.json").write_text(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()

