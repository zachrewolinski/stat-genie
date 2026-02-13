import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the AMTL dataset and relabel columns to their semantic meaning.
    Returns a DataFrame with clear column names and derived variables.
    """
    df = pd.read_csv(csv_path)

    # Relabel columns using the semantic descriptions in info.json
    df = df.rename(
        columns={
            "sockets": "tooth_class",       # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",     # Unique specimen identifier
            "genus": "missing_count",       # Number of missing teeth of given class
            "age": "num_sockets",           # Number of observable sockets
            "pop": "age_at_death",          # Estimated age at death
            "num_amtl": "age_uncertainty",  # Uncertainty in age at death
            "stdev_age": "sex_estimate",    # Estimate of sex (0–1 scale)
            "tooth_class": "genus",         # Genus: Homo sapiens, Pan, Papio, Pongo
            "specimen": "region",           # Region / population
        }
    )

    # Restrict to the genera of interest
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Basic derived quantities
    df["missing_count"] = df["missing_count"].astype(float)
    df["num_sockets"] = df["num_sockets"].astype(float)
    # Guard against any accidental zeros in denominators
    df = df[df["num_sockets"] > 0].copy()

    df["prop_missing"] = df["missing_count"] / df["num_sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Center/scale some covariates lightly for numerical stability (not required conceptually)
    df["age_at_death"] = df["age_at_death"].astype(float)
    df["sex_estimate"] = df["sex_estimate"].astype(float)

    return df


def summarize_group_differences(df: pd.DataFrame) -> dict:
    """
    Compute descriptive AMTL frequencies for humans vs non-human primates.
    Returns a dictionary with aggregate proportions.
    """
    agg = df.groupby("genus").agg(
        total_missing=("missing_count", "sum"),
        total_sockets=("num_sockets", "sum"),
    )
    agg["prop_missing"] = agg["total_missing"] / agg["total_sockets"]

    human_prop = float(agg.loc["Homo sapiens", "prop_missing"])

    nonhuman_mask = agg.index != "Homo sapiens"
    nonhuman_missing = float(agg.loc[nonhuman_mask, "total_missing"].sum())
    nonhuman_sockets = float(agg.loc[nonhuman_mask, "total_sockets"].sum())
    nonhuman_prop = nonhuman_missing / nonhuman_sockets

    return {
        "human_prop": human_prop,
        "nonhuman_prop": nonhuman_prop,
        "by_genus": agg["prop_missing"].to_dict(),
    }


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for proportion of missing teeth with a human indicator
    and covariates age_at_death, sex_estimate, and tooth_class.
    Uses num_sockets as the binomial trial count via freq_weights.
    """
    formula = "prop_missing ~ is_human + age_at_death + sex_estimate + C(tooth_class)"
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result, formula


def compute_marginal_effect(result, df: pd.DataFrame, formula: str) -> float:
    """
    Compute the average marginal effect of being human on the probability
    of AMTL, holding the distribution of covariates fixed.
    """
    df_human = df.copy()
    df_nonhuman = df.copy()
    df_human["is_human"] = 1
    df_nonhuman["is_human"] = 0

    _, X_h = patsy.dmatrices(formula, df_human, return_type="dataframe")
    _, X_n = patsy.dmatrices(formula, df_nonhuman, return_type="dataframe")

    pred_h = result.predict(X_h)
    pred_n = result.predict(X_n)

    avg_diff = float((pred_h - pred_n).mean())
    return avg_diff


def derive_conclusion(descriptive: dict, result, avg_diff: float) -> dict:
    """
    Combine descriptive and model-based evidence into the required
    response, strength, confidence, and explanation.
    """
    human_prop = descriptive["human_prop"]
    nonhuman_prop = descriptive["nonhuman_prop"]

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Direction based on both descriptive and model estimates
    direction_positive = (human_prop > nonhuman_prop) and (avg_diff > 0)
    if direction_positive:
        response = "Yes"
    else:
        response = "No"

    # Effect magnitude for scaling strength (absolute difference in probabilities)
    effect_size = abs(avg_diff)

    # Map effect size (0–0.25+) to 0–100
    strength = int(max(0.0, min(100.0, effect_size / 0.25 * 100.0)))

    # Confidence based mainly on p-value, with a modest boost if descriptive and model agree
    if pval < 1e-4:
        base_conf = 90
    elif pval < 1e-3:
        base_conf = 85
    elif pval < 1e-2:
        base_conf = 80
    elif pval < 5e-2:
        base_conf = 70
    elif pval < 1e-1:
        base_conf = 60
    else:
        base_conf = 45

    if direction_positive:
        base_conf += 5

    confidence = int(max(0, min(100, base_conf)))

    explanation = (
        "I analyzed antemortem tooth loss (AMTL) using binomial regression on the counts "
        "of missing teeth and observable sockets for modern humans (Homo sapiens) and three "
        "non-human primate genera (Pan, Pongo, Papio). I first computed aggregate AMTL "
        f"frequencies: humans had an estimated proportion of missing teeth of "
        f"{human_prop:.3f}, while non-human primates combined had {nonhuman_prop:.3f}. "
        "I then fit a generalized linear model with a binomial family where the response "
        "was the proportion of missing teeth, the trial count was the number of sockets, "
        "and predictors included a binary indicator for humans versus non-humans, "
        "estimated age at death, sex estimate, and tooth class (anterior, posterior, premolar). "
        f"The coefficient for the human indicator corresponded to an odds ratio of {odds_ratio:.2f}, "
        f"with a p-value of {pval:.3g}, and the average marginal effect of being human on the "
        f"probability of AMTL across the observed covariate distribution was {avg_diff:.3f}. "
        "Based on the direction and magnitude of this effect, along with its statistical "
        "significance and the descriptive differences in AMTL frequencies, I concluded "
        f"a '{response}' answer to the research question, with a strength of {strength} out of 100 "
        f"and a confidence of {confidence} out of 100."
    )

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    df = load_and_prepare_data("amtl.csv")
    descriptive = summarize_group_differences(df)
    result, formula = fit_binomial_model(df)
    avg_diff = compute_marginal_effect(result, df, formula)
    conclusion = derive_conclusion(descriptive, result, avg_diff)

    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

