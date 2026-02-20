import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load AMTL dataset and prepare variables for modeling."""
    df = pd.read_csv(csv_path)

    # Rename columns to meaningful names based on info.json metadata
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_est",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: keep valid rows only
    df = df[
        (df["sockets"].notna())
        & (df["missing"].notna())
        & (df["age"].notna())
        & (df["sex_est"].notna())
    ].copy()

    # Ensure integer counts and valid ranges
    df["sockets"] = df["sockets"].astype(int)
    df["missing"] = df["missing"].astype(int)
    df = df[(df["sockets"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets"])]

    # Indicator for modern humans vs non-human primates
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Keep only relevant genera (Homo sapiens, Pan, Papio, Pongo)
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus"].isin(valid_genera)].copy()

    return df


def expand_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand specimen-level counts to tooth-level binary outcomes."""
    rows = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        missing = int(row["missing"])

        # Skip rows with zero sockets (already filtered, but keep safe)
        if sockets <= 0:
            continue

        # 1 for missing (AMTL), 0 for present tooth
        amtl = np.concatenate(
            [np.ones(missing, dtype=int), np.zeros(sockets - missing, dtype=int)]
        )

        tooth_df = pd.DataFrame(
            {
                "amtl": amtl,
                "human": row["human"],
                "age": row["age"],
                "sex_est": row["sex_est"],
                "tooth_class": row["tooth_class"],
            }
        )
        rows.append(tooth_df)

    expanded = pd.concat(rows, ignore_index=True)
    return expanded


def fit_binomial_model(expanded: pd.DataFrame):
    """Fit binomial regression of AMTL on human status and covariates."""
    formula = "amtl ~ human + age + sex_est + C(tooth_class)"
    model = smf.glm(formula=formula, data=expanded, family=sm.families.Binomial())
    result = model.fit()
    return result


def evaluate_human_effect(result) -> Tuple[str, str]:
    """
    Evaluate whether modern humans have higher AMTL frequency.

    Returns:
        response: "Yes" or "No"
        explanation: textual summary of evidence
    """
    # Extract coefficient, p-value, and confidence interval for human effect
    coef_human = float(result.params.get("human", np.nan))
    pval_human = float(result.pvalues.get("human", np.nan))
    ci = result.conf_int().loc["human"].tolist()
    ci_low, ci_high = float(ci[0]), float(ci[1])

    # Average marginal effect: compare predicted probabilities with human=1 vs 0
    expanded = result.model.data.frame.copy()
    expanded_human = expanded.copy()
    expanded_human["human"] = 1
    expanded_nonhuman = expanded.copy()
    expanded_nonhuman["human"] = 0

    pred_human = float(result.predict(expanded_human).mean())
    pred_nonhuman = float(result.predict(expanded_nonhuman).mean())
    diff = pred_human - pred_nonhuman

    # Decide Yes/No based on sign and significance of the human coefficient
    # Use a conventional alpha = 0.05
    is_higher = (coef_human > 0) and (pval_human < 0.05)
    response = "Yes" if is_higher else "No"

    direction = "higher" if coef_human > 0 else "lower"

    explanation = (
        "We modeled the probability that an individual tooth was missing "
        "(antemortem tooth loss, AMTL) using a binomial (logistic) regression "
        "with predictors for modern human vs. non-human primate status, age at death, "
        "estimated sex, and tooth class (anterior, premolar, posterior). "
        f"The coefficient for modern humans (vs. non-human genera Pan, Papio, Pongo) "
        f"was {coef_human:.3f} on the log-odds scale with p-value {pval_human:.3g} "
        f"and a 95% confidence interval from {ci_low:.3f} to {ci_high:.3f}. "
        f"Model-based predicted AMTL frequencies, holding age, sex, and tooth class constant, "
        f"were {pred_human:.3%} for modern humans and {pred_nonhuman:.3%} for non-human primates, "
        f"a difference of {diff:.3%} ({direction} in humans). "
        "Based on the sign and statistical significance of the modern human coefficient, "
        f"we therefore conclude that modern humans do{' ' if is_higher else ' not '}exhibit "
        "higher frequencies of AMTL than non-human primates after accounting for age, sex, "
        "and tooth class."
    )

    return response, explanation


def main():
    df = load_and_prepare_data("amtl.csv")

    if df.empty:
        response = "No"
        explanation = (
            "The dataset contained no valid observations after basic cleaning, "
            "so we could not estimate whether humans have higher AMTL frequencies "
            "than non-human primates."
        )
    else:
        expanded = expand_tooth_level(df)
        if expanded.empty:
            response = "No"
            explanation = (
                "After expanding specimen-level counts to tooth-level data, "
                "no valid teeth remained for analysis, so we could not estimate "
                "differences in AMTL between humans and non-human primates."
            )
        else:
            result = fit_binomial_model(expanded)
            response, explanation = evaluate_human_effect(result)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()

