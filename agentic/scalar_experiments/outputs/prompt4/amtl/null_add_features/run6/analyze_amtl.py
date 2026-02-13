import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load AMTL data and apply basic cleaning."""
    df = pd.read_csv(csv_path)

    # Keep only columns needed for the core question
    cols_needed = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
    ]
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in data: {missing_cols}")

    df = df[cols_needed].copy()

    # Remove rows with missing key values
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Ensure numeric types
    df["num_amtl"] = pd.to_numeric(df["num_amtl"], errors="coerce")
    df["sockets"] = pd.to_numeric(df["sockets"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["prob_male"] = pd.to_numeric(df["prob_male"], errors="coerce")

    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male"])

    # Filter for biologically plausible values
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Restrict to focal genera
    focal_genera = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(focal_genera)].copy()

    # Categorical variables
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def summarize_raw_rates(df: pd.DataFrame) -> dict:
    """Compute simple genus-level AMTL rates."""
    grouped = df.groupby("genus").agg(total_amtl=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
    grouped["rate"] = grouped["total_amtl"] / grouped["total_sockets"]
    return grouped["rate"].to_dict()


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a logistic regression for AMTL with genus, age, sex proxy, and tooth class.

    To avoid numerical issues with aggregated binomial modeling, we expand each
    row into per-socket Bernoulli observations and fit a logistic regression
    using the formula interface.
    """
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_amtl = int(row["num_amtl"])
        if sockets <= 0:
            continue
        num_amtl = max(0, min(num_amtl, sockets))
        # First num_amtl sockets are coded as AMTL events (1), remaining as 0
        for i in range(sockets):
            records.append(
                {
                    "amtl_event": 1 if i < num_amtl else 0,
                    "genus": row["genus"],
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "tooth_class": row["tooth_class"],
                }
            )

    long_df = pd.DataFrame(records)
    long_df["genus"] = long_df["genus"].astype("category")
    long_df["tooth_class"] = long_df["tooth_class"].astype("category")

    formula = "amtl_event ~ C(genus, Treatment(reference='Papio')) + age + prob_male + C(tooth_class)"
    model = smf.logit(formula=formula, data=long_df)
    result = model.fit(disp=False)
    return result


def marginal_predicted_probs(result, df: pd.DataFrame) -> dict:
    """
    Compute marginal predicted AMTL probabilities for each genus
    via standardization: for each observation, set genus to g and
    average predicted probabilities.
    """
    base_covariates = df[["age", "prob_male", "tooth_class"]].copy()
    genera = sorted(df["genus"].unique())

    preds = {}
    for g in genera:
        pred_df = base_covariates.copy()
        pred_df["genus"] = g
        # Prediction on probability scale; weights not used here
        probs = result.predict(pred_df)
        preds[str(g)] = float(np.mean(probs))
    return preds


def derive_conclusion(raw_rates: dict, marginal_probs: dict, result) -> tuple[int, str]:
    """
    Use model outputs to answer:
    Do modern humans have higher AMTL frequencies than non-human primates
    after accounting for age, sex, and tooth class?
    """
    # Extract genus-level info
    human_key = "Homo sapiens"
    nhp_keys = [g for g in marginal_probs.keys() if g != human_key]

    if human_key not in marginal_probs or not nhp_keys:
        # If we cannot make the comparison, respond as uncertain.
        explanation = (
            "The available data do not allow a clear comparison of AMTL "
            "between modern humans and non-human primates after adjusting "
            "for age, sex, and tooth class."
        )
        return 50, explanation

    human_marginal = marginal_probs[human_key]
    nhp_marginals = np.array([marginal_probs[k] for k in nhp_keys])
    mean_nhp_marginal = float(np.mean(nhp_marginals))

    human_raw = raw_rates.get(human_key, float("nan"))
    nhp_raw_vals = np.array([raw_rates.get(k, np.nan) for k in nhp_keys], dtype=float)
    mean_nhp_raw = float(np.nanmean(nhp_raw_vals)) if np.isfinite(nhp_raw_vals).any() else float("nan")

    # Pull key model coefficients and p-values
    coefs = result.params
    pvalues = result.pvalues

    # Effect of Homo sapiens vs Papio (reference in model)
    coef_name_human_vs_papio = "C(genus, Treatment(reference='Papio'))[T.Homo sapiens]"
    human_vs_papio_coef = float(coefs.get(coef_name_human_vs_papio, np.nan))
    human_vs_papio_p = float(pvalues.get(coef_name_human_vs_papio, np.nan))

    # Heuristic strength score based on consistency of evidence
    diff_marginal = human_marginal - mean_nhp_marginal
    diff_raw = human_raw - mean_nhp_raw if np.isfinite(human_raw) and np.isfinite(mean_nhp_raw) else np.nan

    # Initialize with neutral
    score = 50

    # Direction: do humans have higher AMTL?
    humans_higher_marginal = diff_marginal > 0
    humans_higher_raw = diff_raw > 0 if np.isfinite(diff_raw) else humans_higher_marginal

    if humans_higher_marginal and humans_higher_raw and human_vs_papio_coef > 0 and human_vs_papio_p < 0.05:
        # Strong, consistent evidence that humans have higher AMTL
        # Map magnitude of marginal difference into [80, 100]
        # Typical AMTL rates are modest; scaling by 0.2 caps extremes.
        base = 80
        magnitude_component = max(0.0, min(20.0, diff_marginal * 200.0))
        score = int(round(base + magnitude_component))
    elif humans_higher_marginal and human_vs_papio_coef > 0 and human_vs_papio_p < 0.1:
        # Moderate evidence in model, directionally consistent
        base = 65
        magnitude_component = max(0.0, min(20.0, diff_marginal * 200.0))
        score = int(round(base + magnitude_component))
    elif humans_higher_marginal and humans_higher_raw:
        # Weak but directionally consistent evidence
        score = 60
    elif not humans_higher_marginal and not humans_higher_raw and human_vs_papio_coef < 0 and human_vs_papio_p < 0.05:
        # Strong evidence that humans do NOT have higher AMTL
        base = 20
        magnitude_component = max(0.0, min(20.0, -diff_marginal * 200.0))
        score = int(round(base - magnitude_component))
        score = max(0, score)
    else:
        # Mixed or inconclusive evidence
        score = 50

    score = int(max(0, min(100, score)))

    # Build explanation text
    lines = []
    lines.append(
        "I modeled the probability of antemortem tooth loss (AMTL) at the tooth-class level "
        "using a binomial regression with the number of missing teeth as the outcome and the "
        "number of observable sockets as the binomial denominator."
    )
    lines.append(
        "The predictors included taxonomic genus (Homo sapiens vs. Pan, Pongo, Papio), "
        "estimated age at death, a probabilistic sex indicator (probability of being male), "
        "and tooth class (anterior, premolar, posterior)."
    )

    if np.isfinite(human_raw) and np.isfinite(mean_nhp_raw):
        lines.append(
            f"Raw AMTL frequencies (total missing teeth divided by total sockets) were "
            f"approximately {human_raw:.3f} for modern humans and {mean_nhp_raw:.3f} on average "
            f"across the three non-human primate genera."
        )

    lines.append(
        f"After fitting the binomial model and standardizing over the observed distributions of age, "
        f"sex, and tooth class, the marginal predicted probability of AMTL was about "
        f"{human_marginal:.3f} for modern humans and {mean_nhp_marginal:.3f} on average for the "
        f"non-human primate genera."
    )

    if np.isfinite(human_vs_papio_coef) and np.isfinite(human_vs_papio_p):
        direction = "higher" if human_vs_papio_coef > 0 else "lower"
        lines.append(
            "In the regression model with Papio as the reference genus, the coefficient for "
            f"modern humans relative to Papio was {human_vs_papio_coef:.3f} on the log-odds scale "
            f"({direction} AMTL for humans), with a p-value of {human_vs_papio_p:.3f}."
        )

    if humans_higher_marginal and humans_higher_raw:
        lines.append(
            "Both the raw genus-level AMTL frequencies and the adjusted marginal probabilities "
            "indicate that modern humans experience AMTL at somewhat higher rates than the "
            "non-human primate genera considered, even after accounting for age, sex, and tooth class."
        )
    elif not humans_higher_marginal and not humans_higher_raw:
        lines.append(
            "Both the raw frequencies and the adjusted marginal probabilities suggest that modern humans "
            "do not have higher AMTL than the non-human primate genera once age, sex, and tooth class "
            "are taken into account."
        )
    else:
        lines.append(
            "The raw frequencies and adjusted marginal probabilities do not fully agree, so the evidence "
            "for systematically higher AMTL in modern humans relative to non-human primates is mixed."
        )

    if score >= 60:
        lines.append(
            f"Overall, these results provide reasonably consistent evidence that modern humans have "
            f"higher AMTL frequencies than non-human primates after adjustment, corresponding to a "
            f"confidence score of {score} on a 0–100 scale (where higher values indicate stronger "
            f"support for a 'Yes' answer)."
        )
    elif score <= 40:
        lines.append(
            f"Overall, these results suggest that modern humans do not have higher AMTL frequencies "
            f"than non-human primates after adjustment, corresponding to a confidence score of "
            f"{score} on a 0–100 scale (where lower values indicate stronger support for a 'No' answer)."
        )
    else:
        lines.append(
            f"Because some indicators are inconsistent, the evidence is inconclusive, corresponding to "
            f"a confidence score of {score} on a 0–100 scale (values near 50 indicate ambiguity)."
        )

    explanation = " ".join(lines)
    return score, explanation


def main() -> None:
    data_path = Path("amtl.csv")
    df = load_and_clean_data(data_path)

    raw_rates = summarize_raw_rates(df)
    model_result = fit_binomial_model(df)
    marginal_probs = marginal_predicted_probs(model_result, df)

    score, explanation = derive_conclusion(raw_rates, marginal_probs, model_result)

    conclusion = {"response": int(score), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
