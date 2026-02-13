import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "sockets": "tooth_class",
            "prob_male": "specimen_id",
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age_est",
            "num_amtl": "age_uncertainty",
            "stdev_age": "sex_est",
            "tooth_class": "genus",
            "specimen": "region",
        }
    )

    df = df[df["num_sockets"] > 0].copy()
    df["p_missing"] = df["num_missing"] / df["num_sockets"]

    df = df[
        df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])
    ].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["p_missing", "age_est", "sex_est", "tooth_class", "genus"]
    )

    df["age_est"] = df["age_est"].astype(float)
    df["sex_est"] = df["sex_est"].astype(float)
    df["num_sockets"] = df["num_sockets"].astype(float)

    df = df[(df["p_missing"] >= 0.0) & (df["p_missing"] <= 1.0)].copy()

    return df


def fit_model(df: pd.DataFrame):
    formula = (
        'p_missing ~ C(genus, Treatment(reference="Homo sapiens"))'
        " + age_est + sex_est + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(result) -> dict:
    effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{genus}]'
        if term in result.params.index:
            coef = float(result.params[term])
            pval = float(result.pvalues[term])
            effects[genus] = {"coef": coef, "pval": pval}
    return effects


def predicted_missing_probs_by_genus(
    result, df: pd.DataFrame
) -> dict[str, float]:
    probs = {}
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    for genus in genera:
        df_pred = df.copy()
        df_pred["genus"] = genus
        pred = result.predict(df_pred)
        mean_prob = float(
            np.average(pred, weights=df_pred["num_sockets"])
        )
        probs[genus] = mean_prob
    return probs


def decide_conclusion(effects: dict, probs: dict) -> tuple[str, int, str]:
    human_prob = probs.get("Homo sapiens", np.nan)
    others = ["Pan", "Papio", "Pongo"]

    higher_than_all = all(
        np.isfinite(human_prob)
        and human_prob > probs.get(genus, -np.inf)
        for genus in others
    )

    pvals = [
        effects[g]["pval"]
        for g in others
        if g in effects and np.isfinite(effects[g]["pval"])
    ]
    min_p = min(pvals) if pvals else np.nan

    if higher_than_all and np.isfinite(min_p) and min_p < 0.001:
        response = "Yes"
        confidence = 95
    elif higher_than_all and np.isfinite(min_p) and min_p < 0.05:
        response = "Yes"
        confidence = 85
    elif (not higher_than_all) and np.isfinite(min_p) and min_p < 0.05:
        response = "No"
        confidence = 85
    else:
        response = "No" if not higher_than_all else "Yes"
        confidence = 60

    lines = []
    lines.append(
        "Binomial regression of AMTL proportion (missing teeth / observable"
        " sockets) per specimen and tooth class was fit with a logit link,"
        " using Homo sapiens as the reference genus and controlling for"
        " estimated age at death, sex estimate, and tooth class"
        " (anterior/posterior/premolar)."
    )

    if effects:
        for genus, stats in effects.items():
            direction = (
                "lower"
                if stats["coef"] < 0
                else "higher"
                if stats["coef"] > 0
                else "similar"
            )
            lines.append(
                f" Relative to modern humans, {genus} shows {direction}"
                f" AMTL log-odds (coefficient={stats['coef']:.3f},"
                f" p-value={stats['pval']:.3g})."
            )

    if probs:
        prob_parts = []
        for genus, prob in probs.items():
            prob_parts.append(f"{genus}≈{prob*100:.1f}%")
        lines.append(
            " Model-based estimated AMTL frequencies (per tooth socket at"
            " average covariates) are: " + ", ".join(prob_parts) + "."
        )

    if response == "Yes":
        lines.append(
            " Together, these results indicate that modern humans have higher"
            " AMTL frequencies than the non-human primate genera after"
            " accounting for age, sex, and tooth class."
        )
    else:
        lines.append(
            " Taken together, these results do not provide strong evidence"
            " that modern humans have higher AMTL frequencies than all"
            " non-human primate genera once age, sex, and tooth class are"
            " accounted for."
        )

    explanation = "".join(lines)
    return response, confidence, explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)

    result = fit_model(df)
    effects = summarize_genus_effects(result)
    probs = predicted_missing_probs_by_genus(result, df)

    response, confidence, explanation = decide_conclusion(effects, probs)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

