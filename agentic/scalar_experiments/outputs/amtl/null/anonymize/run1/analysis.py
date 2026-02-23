import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age_at_death",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: keep only valid rows
    df = df[(df["n_sockets"] > 0) & (df["n_missing"] >= 0) & (df["n_missing"] <= df["n_sockets"])]
    if df.empty:
        raise ValueError("No valid rows remaining after filtering.")

    # Outcome as proportion with binomial variance weights
    df["missing_prop"] = df["n_missing"] / df["n_sockets"]

    # Ensure Homo sapiens is present and set as reference
    if "Homo sapiens" not in df["genus"].unique():
        raise ValueError("Expected genus 'Homo sapiens' not found in data.")

    # Choose a common tooth class for marginal predictions
    if not df["tooth_class"].empty:
        common_tooth_class = df["tooth_class"].mode(dropna=True).iat[0]
    else:
        common_tooth_class = "Posterior"

    # Binomial GLM with Homo sapiens as reference genus,
    # controlling for age, sex, and tooth class.
    formula = (
        "missing_prop ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + C(tooth_class)"
        " + age_at_death"
        " + sex_estimate"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["n_sockets"],
    ).fit()

    # Extract genus effects (log-odds differences vs Homo sapiens)
    genus_effects = {}
    for name, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
        if name.startswith("C(genus"):
            # name format: C(genus, Treatment(reference='Homo sapiens'))[T.Pan]
            genus_name = name.split("]")[0].split("[T.")[-1]
            genus_effects[genus_name] = {"coef": float(coef), "pval": float(pval)}

    non_human_genera = [g for g in df["genus"].unique() if g != "Homo sapiens"]

    # Determine whether humans have higher AMTL than each non-human genus
    # In this coding, a negative coefficient for genus G means logit(p_G) < logit(p_Homo),
    # i.e., Homo sapiens has higher AMTL than genus G.
    evidence_counts = {"significant_higher": 0, "n_nonhuman": len(non_human_genera)}

    per_genus_conclusions = []
    for g in sorted(non_human_genera):
        effect = genus_effects.get(g)
        if effect is None:
            # If the genus was used as reference for some reason, fall back to predictions.
            conclusion = {
                "genus": g,
                "interpretation": "effect not directly estimated; using predicted probabilities only",
            }
        else:
            coef = effect["coef"]
            pval = effect["pval"]
            if coef < 0 and pval < 0.05:
                evidence_counts["significant_higher"] += 1
                interp = "Humans have significantly higher AMTL than this genus (p < 0.05)."
            elif coef < 0 and pval < 0.10:
                interp = "Humans tend to have higher AMTL than this genus (0.05 <= p < 0.10)."
            else:
                interp = "No statistically clear evidence that humans have higher AMTL than this genus."

            conclusion = {
                "genus": g,
                "coef_vs_humans": coef,
                "p_value": pval,
                "interpretation": interp,
            }

        per_genus_conclusions.append(conclusion)

    # Compute adjusted predicted AMTL probabilities for each genus at typical covariate values
    typical_age = float(df["age_at_death"].median())
    typical_sex = float(df["sex_estimate"].mean())

    pred_rows = []
    for g in sorted(df["genus"].unique()):
        pred_rows.append(
            {
                "genus": g,
                "tooth_class": common_tooth_class,
                "age_at_death": typical_age,
                "sex_estimate": typical_sex,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    pred_df["pred_missing_prop"] = model.predict(pred_df)

    genus_predictions = {
        row["genus"]: float(row["pred_missing_prop"]) for _, row in pred_df.iterrows()
    }

    # Determine overall answer and Likert-scale response
    n_sig = evidence_counts["significant_higher"]
    n_total = evidence_counts["n_nonhuman"]

    if n_total == 0:
        response_score = 50
        overall_conclusion = (
            "The dataset contains only modern humans, so a comparison to non-human primates "
            "is not possible. The answer is therefore uncertain."
        )
    else:
        # Map strength of evidence to a 0–100 scale.
        if n_sig == n_total:
            response_score = 90
            overall_conclusion = (
                "Yes. Binomial regression controlling for age, sex, and tooth class "
                "indicates that modern humans have significantly higher frequencies of "
                "antemortem tooth loss than all sampled non-human primate genera."
            )
        elif n_sig >= 1:
            response_score = 70
            overall_conclusion = (
                "Mostly yes. Humans show significantly higher AMTL than some non-human genera "
                "after adjustment, but the evidence is weaker or non-significant for at least "
                "one genus."
            )
        else:
            response_score = 30
            overall_conclusion = (
                "No clear evidence. After accounting for age, sex, and tooth class, the model "
                "does not show consistent statistically significant differences indicating that "
                "humans have higher AMTL than non-human primates."
            )

    # Build detailed explanation string for the conclusion file
    explanation_parts = []
    explanation_parts.append(
        "I fit a binomial generalized linear model (logistic regression) to the AMTL dataset, "
        "using the number of missing teeth out of the number of observable sockets as the "
        "outcome. The model included genus (with Homo sapiens as the reference category), "
        "tooth class (anterior, premolar, posterior), estimated age at death, and a "
        "continuous sex estimate as predictors, with the number of sockets used as "
        "binomial variance weights."
    )

    # Add per-genus statistical summary
    genus_summaries = []
    for item in per_genus_conclusions:
        g = item["genus"]
        pred = genus_predictions.get(g)
        if "coef_vs_humans" in item:
            coef = item["coef_vs_humans"]
            pval = item["p_value"]
            genus_summaries.append(
                f"For genus {g}, the log-odds difference relative to humans is {coef:.3f} "
                f"(p = {pval:.3f}); the model-predicted proportion of teeth missing under "
                f"typical conditions is approximately {pred:.3f}."
            )
        else:
            genus_summaries.append(
                f"For genus {g}, the effect relative to humans could not be directly read "
                f"from the model coefficients; the model-predicted proportion of teeth "
                f"missing under typical conditions is approximately {pred:.3f}."
            )

    # Add human reference prediction
    human_pred = genus_predictions.get("Homo sapiens")
    if human_pred is not None:
        genus_summaries.append(
            f"For Homo sapiens, the model-predicted proportion of teeth missing under "
            f"typical conditions is approximately {human_pred:.3f}."
        )

    explanation_parts.append(" ".join(genus_summaries))
    explanation_parts.append(overall_conclusion)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response_score), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

