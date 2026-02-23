import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Reconstruct variable meanings based on metadata and column samples
    # - df["genus"]        : count of missing teeth in this record
    # - df["age"]          : number of observable sockets
    # - df["pop"]          : estimated age at death
    # - df["stdev_age"]    : estimated probability of male (sex estimate)
    # - df["tooth_class"]  : actual genus (Homo sapiens, Pan, Papio, Pongo)
    # - df["sockets"]      : tooth class (Anterior / Posterior / Premolar)

    df = df.copy()
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]
    df["age_est"] = df["pop"].astype(float)
    df["sex_est"] = df["stdev_age"].astype(float)
    df["genus_cat"] = df["tooth_class"].astype("category")
    df["tooth_class_cat"] = df["sockets"].astype("category")

    # Basic cleaning: remove rows with invalid socket counts or missing key fields
    df = df[
        (df["num_sockets"] > 0)
        & df["missing_prop"].notna()
        & df["genus_cat"].notna()
        & df["tooth_class_cat"].notna()
        & df["age_est"].notna()
        & df["sex_est"].notna()
    ].copy()

    # Descriptive AMTL frequencies by genus
    genus_summary = (
        df.groupby("genus_cat")
        .agg(
            total_missing=("num_missing", "sum"),
            total_sockets=("num_sockets", "sum"),
            n_records=("num_missing", "size"),
        )
    )
    genus_summary["prop_missing"] = (
        genus_summary["total_missing"] / genus_summary["total_sockets"]
    )

    # Fit binomial regression model:
    # response: missing_prop with binomial family and num_sockets as trial weights
    # predictors: genus (Homo sapiens as reference), age at death, sex estimate, tooth class
    model = None
    fit_successful = False
    effects: dict[str, dict[str, float]] = {}

    try:
        formula = (
            "missing_prop ~ "
            "C(genus_cat, Treatment(reference='Homo sapiens')) + "
            "age_est + sex_est + C(tooth_class_cat)"
        )
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["num_sockets"],
        ).fit()
        fit_successful = True

        params = model.params
        pvalues = model.pvalues
        for genus in ["Pan", "Papio", "Pongo"]:
            key = f"C(genus_cat, Treatment(reference='Homo sapiens'))[T.{genus}]"
            if key in params:
                effects[genus] = {
                    "coef": float(params[key]),
                    "p": float(pvalues[key]),
                }
    except Exception as exc:  # noqa: BLE001
        # Fall back to descriptive-only conclusion if the model fails
        fit_successful = False
        model = None
        effects = {"_error": {"message": str(exc)}}

    # Map statistical evidence to a 0–100 Likert-scale response
    def map_evidence_to_score(
        genus_effects: dict[str, dict[str, float]],
        summary: pd.DataFrame,
    ) -> int:
        # If model failed, fallback to descriptive comparison of proportions
        if not genus_effects or "_error" in genus_effects:
            # Compare pooled proportion for Homo sapiens vs pooled non-human genera
            if "Homo sapiens" not in summary.index:
                return 50

            human = summary.loc["Homo sapiens"]
            nonhuman = summary.drop(index=["Homo sapiens"])
            total_missing_nh = nonhuman["total_missing"].sum()
            total_sockets_nh = nonhuman["total_sockets"].sum()
            prop_h = human["prop_missing"]
            prop_nh = (
                total_missing_nh / total_sockets_nh
                if total_sockets_nh > 0
                else np.nan
            )

            if not np.isfinite(prop_nh):
                return 50

            diff = float(prop_h - prop_nh)
            # Simple heuristic: small differences yield moderate scores
            if diff > 0:
                if diff > 0.10:
                    return 75
                if diff > 0.05:
                    return 65
                return 55
            if diff < 0:
                if diff < -0.10:
                    return 25
                if diff < -0.05:
                    return 35
                return 45
            return 50

        # Use model-based effects: negative coef => non-human genus has lower AMTL
        partial_scores: list[float] = []
        for genus in ["Pan", "Papio", "Pongo"]:
            eff = genus_effects.get(genus)
            if eff is None:
                continue
            coef = eff["coef"]
            p = eff["p"]

            # Start at neutral 0.5, move toward 1 for evidence that
            # Homo sapiens has higher AMTL, or toward 0 for the opposite.
            score = 0.5
            if coef < 0:  # Homo sapiens higher AMTL than this genus
                if p < 0.001:
                    score = 1.0
                elif p < 0.01:
                    score = 0.9
                elif p < 0.05:
                    score = 0.8
                else:
                    score = 0.6
            elif coef > 0:  # Homo sapiens lower AMTL than this genus
                if p < 0.001:
                    score = 0.0
                elif p < 0.01:
                    score = 0.1
                elif p < 0.05:
                    score = 0.2
                else:
                    score = 0.4
            else:
                score = 0.5

            partial_scores.append(score)

        if not partial_scores:
            return 50

        mean_score = float(np.mean(partial_scores))
        mean_score = min(max(mean_score, 0.0), 1.0)
        return int(round(mean_score * 100))

    response_value = map_evidence_to_score(effects, genus_summary)

    # Build human-readable explanation
    lines: list[str] = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primate "
        "genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?",
    )
    lines.append(
        f"Dataset: {len(df)} specimen–tooth-class records with counts of missing "
        "teeth, observable sockets, estimated age at death, sex estimate, "
        "tooth class (anterior/posterior/premolar), and genus.",
    )

    lines.append(
        "Descriptive AMTL frequencies by genus (missing teeth / observable sockets):",
    )
    for genus, row in genus_summary.sort_index().iterrows():
        lines.append(
            f" - {genus}: {row['total_missing']:.0f} missing of "
            f"{row['total_sockets']:.0f} sockets "
            f"(proportion {row['prop_missing']:.3f}, {row['n_records']} records).",
        )

    if fit_successful and model is not None and effects:
        lines.append(
            "Inferential analysis: A binomial regression (logit link) was fit "
            "to the proportion of missing teeth per record, using the number "
            "of observable sockets as binomial trial weights. Predictors were "
            "genus (Homo sapiens as the reference category), estimated age at "
            "death, sex estimate, and tooth class.",
        )
        for genus in ["Pan", "Papio", "Pongo"]:
            eff = effects.get(genus)
            if eff is None:
                continue
            direction = "lower" if eff["coef"] < 0 else "higher"
            lines.append(
                f" - Relative to Homo sapiens, {genus} shows {direction} log-odds "
                f"of AMTL (coefficient {eff['coef']:.3f}, p = {eff['p']:.3g}).",
            )
    else:
        lines.append(
            "Inferential analysis: The planned binomial regression could not be "
            "fit reliably; conclusions rely on pooled genus-level AMTL "
            "proportions instead.",
        )

    if response_value > 55:
        verdict = (
            "Overall, the descriptive and regression results provide evidence "
            "that modern humans experience higher frequencies of AMTL than the "
            "non-human primate genera examined, even after adjusting for age, "
            "sex, and tooth class."
        )
    elif response_value < 45:
        verdict = (
            "Overall, the available evidence suggests that modern humans do not "
            "have higher frequencies of AMTL than the non-human primate genera "
            "when age, sex, and tooth class are taken into account."
        )
    else:
        verdict = (
            "Overall, the evidence is mixed and does not strongly support a "
            "difference in AMTL frequencies between modern humans and the "
            "non-human primate genera once age, sex, and tooth class are "
            "controlled."
        )

    lines.append(verdict)
    lines.append(
        f"On a 0–100 Likert scale, where higher values indicate stronger support "
        f"for the statement that modern humans have higher AMTL frequencies than "
        f"non-human primates, the response value is {response_value}.",
    )

    explanation = "\n".join(lines)

    conclusion = {
        "response": int(response_value),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

