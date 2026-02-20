import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age_est",
            "feature6": "age_unc",
            "feature7": "sex_est",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Ensure numeric types where appropriate
    numeric_cols = ["missing", "sockets", "age_est", "age_unc", "sex_est"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_n = len(df)

    # Filter out rows with invalid or impossible counts for a binomial model
    valid_mask = (df["sockets"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets"])
    df_valid = df.loc[valid_mask].copy()
    used_n = len(df_valid)

    # Proportion of missing teeth in each observation
    df_valid["missing_prop"] = df_valid["missing"] / df_valid["sockets"]

    # Categorical predictors
    df_valid["tooth_class"] = df_valid["tooth_class"].astype("category")
    df_valid["genus"] = df_valid["genus"].astype("category")

    # Make sure Homo sapiens is present as a reference level
    genera = list(df_valid["genus"].cat.categories)
    if "Homo sapiens" not in genera:
        explanation = (
            "The filtered dataset does not contain any observations for Homo sapiens, "
            "so it is impossible to compare AMTL frequencies between humans and non-human primates."
        )
        result = {"response": 50, "explanation": explanation}
        with open("conclusion.txt", "w") as f:
            json.dump(result, f)
        return

    # Fit binomial regression (logit link) with Homo sapiens as the reference genus.
    # We model the proportion of missing teeth with the number of observable sockets
    # as frequency weights to respect different denominators.
    formula = (
        'missing_prop ~ C(genus, Treatment(reference="Homo sapiens")) '
        "+ age_est + sex_est + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df_valid,
        family=sm.families.Binomial(),
        freq_weights=df_valid["sockets"],
    ).fit()

    # Model-based predicted mean AMTL frequency for each genus,
    # marginalizing over the observed distribution of age, sex, and tooth class.
    genera = list(df_valid["genus"].cat.categories)
    predicted_means: dict[str, float] = {}
    for g in genera:
        temp = df_valid.copy()
        temp["genus"] = g
        preds = model.predict(temp)
        predicted_means[g] = float(preds.mean())

    # Extract genus effects relative to Homo sapiens
    params = model.params
    pvalues = model.pvalues
    genus_effects: dict[str, dict[str, float]] = {}

    for g in genera:
        if g == "Homo sapiens":
            continue
        param_name = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
        if param_name in params.index:
            genus_effects[g] = {
                "coef": float(params[param_name]),
                "p_value": float(pvalues[param_name]),
            }

    homo_mean = predicted_means.get("Homo sapiens", np.nan)
    other_means = [predicted_means[g] for g in genera if g != "Homo sapiens"]
    mean_others = float(np.mean(other_means)) if other_means else np.nan
    overall_diff = homo_mean - mean_others if not np.isnan(homo_mean) and not np.isnan(mean_others) else np.nan

    num_genera = len(other_means)
    n_higher = sum(homo_mean > predicted_means[g] for g in genera if g != "Homo sapiens")
    n_sig = sum(
        eff["coef"] < 0 and eff["p_value"] < 0.05
        for g, eff in genus_effects.items()
    )

    sig_fraction = n_sig / num_genera if num_genera > 0 else 0.0

    # Convert evidence into a 0-100 Likert-style response.
    # Positive overall_diff means higher AMTL in Homo sapiens.
    if np.isnan(overall_diff) or num_genera == 0:
        score = 50
    elif overall_diff <= 0:
        # Humans have equal or lower AMTL than non-human primates.
        if overall_diff < -0.02 and sig_fraction > 0.5:
            score = 5
        else:
            score = 20
    else:
        # Humans have higher AMTL than non-human primates.
        if overall_diff > 0.10:
            magnitude_factor = 1.0
        elif overall_diff > 0.05:
            magnitude_factor = 0.8
        elif overall_diff > 0.02:
            magnitude_factor = 0.6
        else:
            magnitude_factor = 0.4

        score = int(60 + 40 * magnitude_factor * (0.5 + 0.5 * sig_fraction))

    # Clip score to [0, 100]
    score = int(max(0, min(100, score)))

    # Build explanation string
    lines: list[str] = []
    lines.append(
        "I modeled antemortem tooth loss (AMTL) using a binomial regression "
        "for the number of missing teeth out of observable sockets with a logit link."
    )
    lines.append(
        "Predictors included genus (Homo sapiens vs Pan, Papio, Pongo and any other genera), "
        "estimated age at death, sex estimate, and tooth class (anterior, posterior, premolar)."
    )
    lines.append(
        f"The original dataset contained {original_n} rows; after excluding records with "
        "non-positive socket counts or missing teeth greater than observable sockets, "
        f"{used_n} observations remained for modeling."
    )

    # Summarize predicted AMTL frequencies by genus
    pred_parts = []
    for g in genera:
        pred_parts.append(f"{g}: {predicted_means[g] * 100:.1f}%")
    if pred_parts:
        lines.append(
            "Model-based predicted mean AMTL frequencies (missing teeth per socket), "
            "marginalized over age, sex, and tooth class, were: "
            + "; ".join(pred_parts)
            + "."
        )

    # Summarize genus effects relative to humans
    effect_lines = []
    for g, eff in genus_effects.items():
        direction = "lower" if eff["coef"] < 0 else "higher"
        effect_lines.append(
            f"For {g} versus Homo sapiens, the log-odds difference in AMTL is {eff['coef']:.2f} "
            f"(p = {eff['p_value']:.3g}), indicating {direction} odds of AMTL in {g} after "
            "adjusting for age, sex, and tooth class."
        )
    lines.extend(effect_lines)

    if not np.isnan(overall_diff):
        if overall_diff > 0:
            qualitative = "higher"
        elif overall_diff < 0:
            qualitative = "lower"
        else:
            qualitative = "similar"
        lines.append(
            f"On average, Homo sapiens have {qualitative} predicted AMTL frequencies "
            f"than the non-human primate genera, with an absolute difference of "
            f"{overall_diff * 100:.1f} percentage points."
        )

    lines.append(
        "Taking the direction and statistical strength of these effects together, "
        f"I summarize the answer to the question of whether modern humans have higher "
        f"AMTL frequencies than non-human primates (controlling for age, sex, and tooth "
        f"class) with a score of {score} on a 0–100 scale, where 0 is a strong 'No' "
        "and 100 is a strong 'Yes'."
    )

    explanation = " ".join(lines)

    result = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

