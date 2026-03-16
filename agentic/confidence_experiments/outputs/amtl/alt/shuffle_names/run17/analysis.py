import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base = Path(__file__).parent

    info = json.loads((base / "info.json").read_text())
    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(base / "amtl.csv")

    # Rename columns to semantic names based on the metadata and inspection.
    df = df.rename(
        columns={
            "tooth_class": "genus_label",  # Homo sapiens, Pan, Papio, Pongo
            "sockets": "tooth_class",  # Anterior, Posterior, Premolar
            "genus": "num_missing",  # count of missing teeth
            "age": "num_sockets",  # observable sockets
            "pop": "age_est",  # estimated age at death
            "num_amtl": "age_sd",  # age uncertainty
            "stdev_age": "prob_male",  # probability of being male
            "prob_male": "specimen_id",  # specimen identifier
            "specimen": "region",  # geographic/collection region
        }
    )

    # Basic cleaning: keep rows with positive socket counts and valid missing counts.
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0)]
    df = df[df["num_missing"] <= df["num_sockets"]]

    # Descriptive AMTL frequencies by genus.
    genus_group = (
        df.groupby("genus_label")
        .agg(
            total_missing=("num_missing", "sum"),
            total_sockets=("num_sockets", "sum"),
        )
        .assign(prop_missing=lambda g: g["total_missing"] / g["total_sockets"])
    )

    # Prepare data for binomial regression: proportion missing with socket counts as weights.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    formula = "prop_missing ~ C(genus_label) + age_est + prob_male + C(tooth_class)"

    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["num_sockets"],
        ).fit()
    except Exception as exc:  # pragma: no cover - defensive fallback
        # If the model fails (e.g., due to separation), fall back to a descriptive-only conclusion.
        score = 50
        explanation = (
            f"{question}\n\n"
            "The planned binomial regression model could not be fit due to the following "
            f"issue: {exc}. As a result, the conclusion is based only on raw AMTL "
            "frequencies by genus, without full adjustment for age, sex, and tooth class."
        )
        conclusion = {"response": score, "explanation": explanation}
        (base / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))
        return

    params = model.params
    bse = model.bse
    pvalues = model.pvalues

    # Genus levels we care about; restrict to those actually present.
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    present = [g for g in target_genera if g in df["genus_label"].unique()]

    # Collect genus effects (non-human vs Homo sapiens baseline).
    genus_effects: dict[str, dict[str, float]] = {}
    for g in present:
        if g == "Homo sapiens":
            genus_effects[g] = {
                "coef": 0.0,
                "se": float("nan"),
                "p": float("nan"),
                "odds_ratio": 1.0,
            }
        else:
            term = f"C(genus_label)[T.{g}]"
            coef = float(params.get(term, float("nan")))
            se = float(bse.get(term, float("nan")))
            pv = float(pvalues.get(term, float("nan")))
            genus_effects[g] = {
                "coef": coef,
                "se": se,
                "p": pv,
                "odds_ratio": float(np.exp(coef)) if np.isfinite(coef) else float("nan"),
            }

    # Predicted AMTL probabilities at typical covariate values
    median_age = float(df["age_est"].median())
    median_prob_male = float(df["prob_male"].median())
    tooth_classes = df["tooth_class"].unique()

    rows: list[dict[str, object]] = []
    for g in present:
        for tc in tooth_classes:
            rows.append(
                {
                    "genus_label": g,
                    "age_est": median_age,
                    "prob_male": median_prob_male,
                    "tooth_class": tc,
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_df["pred_prob_missing"] = model.predict(pred_df)
    typical_probs = (
        pred_df.groupby("genus_label")["pred_prob_missing"].mean().to_dict()
    )

    human_prob = typical_probs.get("Homo sapiens", float("nan"))
    evidence_details: list[dict[str, float | str]] = []
    human_higher_count = 0
    n_comparisons = 0

    for g in present:
        if g == "Homo sapiens":
            continue
        n_comparisons += 1
        g_prob = typical_probs.get(g, float("nan"))
        diff = human_prob - g_prob
        eff = genus_effects[g]
        homo_higher = diff > 0 and eff["coef"] < 0
        if homo_higher:
            human_higher_count += 1
        evidence_details.append(
            {
                "other_genus": g,
                "human_prob": human_prob,
                "other_prob": g_prob,
                "diff": diff,
                "coef_vs_human": eff["coef"],
                "pvalue": eff["p"],
            }
        )

    # Map evidence to a 0–100 Likert score.
    score = 50.0
    for detail in evidence_details:
        pv = detail["pvalue"]
        diff = detail["diff"]
        coef = detail["coef_vs_human"]

        if not np.isfinite(pv) or not np.isfinite(diff) or not np.isfinite(coef):
            continue

        if diff > 0 and coef < 0:
            # Evidence humans have higher AMTL than this genus.
            if pv < 0.001:
                score += 15
            elif pv < 0.01:
                score += 12
            elif pv < 0.05:
                score += 8
            elif pv < 0.1:
                score += 4
            else:
                score += 2
        elif diff < 0 and coef > 0:
            # Evidence against humans having higher AMTL.
            if pv < 0.001:
                score -= 15
            elif pv < 0.01:
                score -= 12
            elif pv < 0.05:
                score -= 8
            elif pv < 0.1:
                score -= 4
            else:
                score -= 2

    if n_comparisons > 0:
        if human_higher_count == n_comparisons:
            score += 5
        elif human_higher_count == 0:
            score -= 5

    score = int(round(float(np.clip(score, 0.0, 100.0))))

    lines: list[str] = []
    if question:
        lines.append(question)
        lines.append("")

    lines.append("Analytical approach:")
    lines.append(
        "- Computed AMTL frequencies as the number of missing teeth divided by the number "
        "of observable sockets for each specimen and tooth class."
    )
    lines.append(
        "- Fit a binomial logistic regression (GLM with logit link) for AMTL (missing vs. "
        "present teeth) with genus, estimated age at death, sex (probability of being male), "
        "and tooth class as predictors, using socket counts as binomial denominators."
    )
    lines.append(
        "- Treated modern humans (Homo sapiens) as the reference genus and compared non-human "
        "genera (Pan, Papio, Pongo) against this baseline."
    )
    lines.append("")

    lines.append("Descriptive AMTL frequencies by genus (raw proportions):")
    for g in present:
        prop = genus_group.loc[g, "prop_missing"]
        lines.append(f"- {g}: {prop:.3f} missing teeth per socket (overall).")

    lines.append("")
    lines.append("Model-based results (controlling for age, sex, and tooth class):")
    for g in present:
        if g == "Homo sapiens":
            hp = typical_probs.get(g, float("nan"))
            lines.append(
                f"- Homo sapiens: predicted AMTL probability ≈ {hp:.3f} at typical age and sex,"
                " averaged across tooth classes."
            )
        else:
            detail = next(
                (d for d in evidence_details if d["other_genus"] == g),
                None,
            )
            eff = genus_effects[g]
            if detail is not None:
                gp = detail["other_prob"]
                diff = detail["diff"]
                lines.append(
                    f"- {g}: predicted AMTL probability ≈ {gp:.3f}; difference (Homo sapiens − {g}) "
                    f"≈ {diff:.3f}, regression coefficient vs. humans = {eff['coef']:.3f}, "
                    f"p-value = {eff['p']:.4f}."
                )

    lines.append("")
    if score > 60:
        conclusion_phrase = (
            "Overall, the regression indicates that modern humans have higher AMTL frequencies "
            "than the non-human primate genera considered, even after accounting for age, sex, "
            "and tooth class."
        )
    elif score < 40:
        conclusion_phrase = (
            "Overall, the regression does not support the claim that modern humans have higher "
            "AMTL frequencies than the non-human primate genera once age, sex, and tooth class "
            "are taken into account."
        )
    else:
        conclusion_phrase = (
            "Overall, the regression provides only weak or mixed evidence regarding whether "
            "modern humans have higher AMTL frequencies than the non-human primate genera after "
            "accounting for age, sex, and tooth class."
        )
    lines.append(conclusion_phrase)
    lines.append(
        f"The Likert-scale response of {score} (0 = strong 'No', 100 = strong 'Yes') reflects "
        "the direction and statistical strength of these genus effects."
    )

    explanation = "\n".join(lines)

    conclusion = {"response": score, "explanation": explanation}
    (base / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

