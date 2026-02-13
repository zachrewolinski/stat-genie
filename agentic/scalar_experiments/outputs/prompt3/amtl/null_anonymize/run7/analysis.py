import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base = Path(__file__).resolve().parent

    # Load metadata (used for context, not logic)
    info_path = base / "info.json"
    if info_path.exists():
        _ = json.loads(info_path.read_text())

    # Load dataset
    data_path = base / "amtl.csv"
    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: keep rows with valid counts and covariates
    df = df[
        (df["n_sockets"] > 0)
        & df["n_missing"].notnull()
        & df["age"].notnull()
        & df["sex_estimate"].notnull()
        & df["tooth_class"].notnull()
        & df["genus"].notnull()
    ].copy()

    # Response: proportion of missing teeth in a tooth class, with binomial weights
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Encode categorical predictors
    df["tooth_class"] = df["tooth_class"].astype("category")
    df["genus"] = df["genus"].astype("category")

    # Ensure Homo sapiens is the reference genus if present
    if "Homo sapiens" in df["genus"].cat.categories:
        cats = list(df["genus"].cat.categories)
        cats = ["Homo sapiens"] + [c for c in cats if c != "Homo sapiens"]
        df["genus"] = df["genus"].cat.reorder_categories(cats)

    # Binomial regression: AMTL proportion ~ genus + age + sex + tooth class
    formula = "prop_missing ~ genus + age + sex_estimate + tooth_class"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    # Collect effects for non-human genera relative to Homo sapiens
    genus_effects = {}
    for name in params.index:
        if name.startswith("genus["):
            genus_name = name.split("[T.")[-1].rstrip("]")
            genus_effects[genus_name] = {
                "coef": float(params[name]),
                "pvalue": float(pvalues[name]),
                "ci_low": float(conf_int.loc[name, 0]),
                "ci_high": float(conf_int.loc[name, 1]),
            }

    # Determine directional and statistical evidence
    all_negative = all(v["coef"] < 0 for v in genus_effects.values())
    all_sig_05 = all(v["pvalue"] < 0.05 for v in genus_effects.values())
    any_sig_10 = any(v["pvalue"] < 0.1 for v in genus_effects.values())

    if all_negative and all_sig_05:
        response = "Yes"
        strength = 90
        confidence = 85
    elif all_negative and any_sig_10:
        response = "Yes"
        strength = 70
        confidence = 65
    else:
        response = "No"
        num_negative = sum(v["coef"] < 0 for v in genus_effects.values())
        if num_negative >= 2:
            strength = 40
            confidence = 60
        else:
            strength = 70
            confidence = 75

    # Build an explanation based on model and predictions
    lines = []
    lines.append(
        "I modeled the proportion of missing teeth (AMTL) per specimen and tooth class "
        "using a binomial regression with logit link, where the number of observable "
        "sockets provided the binomial trial counts."
    )
    lines.append(
        "The predictors were genus (Homo sapiens, Pan, Papio, Pongo), age at death, "
        "estimated sex, and tooth class (anterior, posterior, premolar), so genus "
        "effects represent differences in AMTL frequencies after accounting for age, "
        "sex, and tooth class."
    )

    # Predicted AMTL probabilities for each genus at typical covariate values
    ref = {
        "age": float(df["age"].median()),
        "sex_estimate": float(df["sex_estimate"].median()),
        "tooth_class": df["tooth_class"].mode()[0],
    }
    genus_levels = list(df["genus"].cat.categories)
    pred_rows = []
    for g in genus_levels:
        row = {"genus": g}
        row.update(ref)
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_res = result.get_prediction(pred_df).summary_frame(alpha=0.05)
    pred_probs = {
        genus_levels[i]: float(pred_res.iloc[i]["mean"]) for i in range(len(genus_levels))
    }

    lines.append(
        "Using the fitted model, I estimated AMTL frequencies for each genus at the "
        "median age, median sex estimate, and the most common tooth class."
    )
    desc_parts = [
        f"{g}: {pred_probs[g]:.3f}" for g in sorted(genus_levels, key=lambda x: pred_probs[x], reverse=True)
    ]
    lines.append(
        "Predicted proportions of missing teeth (higher values indicate more AMTL) by "
        f"genus were: {', '.join(desc_parts)}."
    )

    lines.append(
        "Model coefficients for the non-human genera are interpreted relative to Homo "
        "sapiens; negative coefficients mean lower AMTL odds than humans after "
        "adjusting for age, sex, and tooth class."
    )
    for genus_name, eff in genus_effects.items():
        lines.append(
            f"For {genus_name}, the log-odds difference relative to Homo sapiens was "
            f"{eff['coef']:.3f} (95% CI {eff['ci_low']:.3f} to {eff['ci_high']:.3f}, "
            f"p = {eff['pvalue']:.3g})."
        )

    if response == "Yes":
        lines.append(
            "Because all non-human genera show lower AMTL than Homo sapiens in this "
            "model, with at least moderate statistical support, the analysis indicates "
            "that modern humans have higher AMTL frequencies than Pan, Papio, and Pongo "
            "after accounting for age, sex, and tooth class."
        )
    else:
        lines.append(
            "Because the non-human genera do not all show clearly and consistently lower "
            "AMTL than Homo sapiens with strong statistical support, the data do not "
            "provide convincing evidence that humans universally have higher AMTL "
            "frequencies than Pan, Papio, and Pongo after accounting for age, sex, and "
            "tooth class; I therefore answer 'No' to the research question as posed."
        )

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    conclusion_path = base / "conclusion.txt"
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

