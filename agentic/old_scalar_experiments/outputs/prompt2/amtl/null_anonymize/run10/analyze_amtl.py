import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Rename columns to meaningful names
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_teeth",
            "feature4": "observable_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning and filtering
    df = df.copy()
    df["observable_sockets"] = pd.to_numeric(df["observable_sockets"], errors="coerce")
    df["missing_teeth"] = pd.to_numeric(df["missing_teeth"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex_estimate"] = pd.to_numeric(df["sex_estimate"], errors="coerce")

    df = df.dropna(subset=["observable_sockets", "missing_teeth", "age", "sex_estimate", "genus", "tooth_class"])
    df = df[df["observable_sockets"] > 0]
    df = df[df["missing_teeth"] >= 0]
    df = df[df["missing_teeth"] <= df["observable_sockets"]]

    # Focus on the relevant genera
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(target_genera)].copy()

    if df.empty or "Homo sapiens" not in df["genus"].unique():
        response = "No"
        confidence = 20
        explanation = (
            "The dataset did not contain sufficient observations for Homo sapiens and the "
            "non-human primate genera (Pan, Papio, Pongo) needed to address the research question, "
            "so the hypothesis that modern humans have higher AMTL frequencies could not be evaluated."
        )
        write_conclusion(response, confidence, explanation)
        return

    # Proportion of missing teeth and binomial weights
    df["prop_missing"] = df["missing_teeth"] / df["observable_sockets"]

    # Fit binomial GLM: AMTL proportion ~ genus + age + sex + tooth class
    formula = "prop_missing ~ C(genus) + C(tooth_class) + age + sex_estimate"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable_sockets"],
    )
    result = model.fit()

    # Use the fitted model to obtain adjusted mean AMTL proportions by genus.
    # We do this by setting genus to each level in turn while keeping other covariates as observed.
    genera = sorted(df["genus"].unique())

    # Identify the columns corresponding to genus indicators in the design matrix
    param_names = list(result.params.index)
    genus_param_prefix = "C(genus)[T."
    genus_cols = {
        name[len(genus_param_prefix) : -1]: idx
        for idx, name in enumerate(param_names)
        if name.startswith(genus_param_prefix)
    }

    X_base = result.model.exog.copy()

    adjusted_means = {}
    for g in genera:
        X_g = X_base.copy()
        # Zero out all genus dummy columns
        for idx in genus_cols.values():
            X_g[:, idx] = 0.0
        # Set the appropriate dummy for non-reference genera; reference (Homo sapiens) has all zeros
        if g != "Homo sapiens":
            col_idx = genus_cols.get(g)
            if col_idx is not None:
                X_g[:, col_idx] = 1.0
        eta = X_g @ result.params.values
        mu = 1.0 / (1.0 + np.exp(-eta))
        adjusted_means[g] = float(mu.mean())

    # Approximate uncertainty via simulation of the coefficient vector
    rng = np.random.default_rng(0)
    params_mean = result.params.values
    cov = result.cov_params().values

    n_draws = 2000
    try:
        beta_draws = rng.multivariate_normal(mean=params_mean, cov=cov, size=n_draws)
    except np.linalg.LinAlgError:
        # Fallback: add a small ridge term if the covariance is not positive definite
        jitter = 1e-8 * np.eye(cov.shape[0])
        beta_draws = rng.multivariate_normal(mean=params_mean, cov=cov + jitter, size=n_draws)

    # Precompute design matrices for each genus scenario
    X_by_genus = {}
    for g in genera:
        X_g = X_base.copy()
        for idx in genus_cols.values():
            X_g[:, idx] = 0.0
        if g != "Homo sapiens":
            col_idx = genus_cols.get(g)
            if col_idx is not None:
                X_g[:, col_idx] = 1.0
        X_by_genus[g] = X_g

    # For each draw, compute adjusted mean AMTL for each genus and the probability that Homo sapiens
    # has the highest AMTL frequency among the four genera.
    homo_label = "Homo sapiens"
    other_labels = [g for g in genera if g != homo_label]

    homo_means_draws = np.empty(n_draws)
    other_means_draws = {g: np.empty(n_draws) for g in other_labels}

    for i in range(n_draws):
        beta = beta_draws[i]
        # Homo sapiens
        eta_homo = X_by_genus[homo_label] @ beta
        mu_homo = 1.0 / (1.0 + np.exp(-eta_homo))
        homo_means_draws[i] = mu_homo.mean()
        # Other genera
        for g in other_labels:
            eta_g = X_by_genus[g] @ beta
            mu_g = 1.0 / (1.0 + np.exp(-eta_g))
            other_means_draws[g][i] = mu_g.mean()

    # Probability that Homo sapiens has the highest adjusted AMTL rate
    higher_than_all = np.ones(n_draws, dtype=bool)
    for g in other_labels:
        higher_than_all &= homo_means_draws > other_means_draws[g]
    prob_homo_highest = float(higher_than_all.mean())

    # Point estimates
    adj_homo = adjusted_means.get(homo_label, np.nan)
    max_other = max(adjusted_means[g] for g in other_labels)

    # Decide on the answer and confidence
    if not np.isnan(adj_homo) and adj_homo > max_other and prob_homo_highest > 0.5:
        response = "Yes"
        prob_directional = prob_homo_highest
    else:
        response = "No"
        prob_directional = 1.0 - prob_homo_highest

    confidence = int(max(0, min(100, round(prob_directional * 100))))

    # Build a concise explanation including key adjusted means and the simulation result
    explanation = (
        f"I fitted a binomial regression model for the proportion of missing teeth "
        f"(missing_teeth / observable_sockets) using genus, tooth class, age at death, "
        f"and sex estimate as predictors on {len(df)} observations. After accounting for "
        f"these covariates, the adjusted mean AMTL proportion for Homo sapiens was "
        f"{adj_homo:.3f}, compared to "
        f"{adjusted_means.get('Pan', float('nan')):.3f} for Pan, "
        f"{adjusted_means.get('Papio', float('nan')):.3f} for Papio, and "
        f"{adjusted_means.get('Pongo', float('nan')):.3f} for Pongo. "
        f"Simulating {n_draws} draws from the estimated coefficient covariance matrix, "
        f"Homo sapiens had the highest adjusted AMTL rate among the four genera in "
        f"{prob_homo_highest * 100:.1f}% of draws. This analysis "
        f"{'supports' if response == 'Yes' else 'does not support'} the hypothesis that "
        f"modern humans have higher AMTL frequencies than non-human primates after "
        f"controlling for age, sex, and tooth class."
    )

    write_conclusion(response, confidence, explanation)


def write_conclusion(response: str, confidence: int, explanation: str) -> None:
    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

