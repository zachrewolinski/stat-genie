import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.copy()
    df["missing"] = df["genus"].astype(int)
    df["total_sockets"] = df["age"].astype(int)
    df["age_years"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)
    df["genus_cat"] = df["tooth_class"].astype(str)
    df["tooth_class"] = df["sockets"].astype(str)
    df["specimen_id"] = df["prob_male"].astype(str)

    df = df[(df["total_sockets"] > 0) & (df["missing"] >= 0)]
    df = df[df["missing"] <= df["total_sockets"]]

    expanded = df.loc[df.index.repeat(df["total_sockets"])].copy()
    expanded["tooth_index"] = expanded.groupby(level=0).cumcount()
    expanded["amtl"] = (expanded["tooth_index"] < expanded["missing"]).astype(int)

    return expanded


def fit_logistic_model(expanded: pd.DataFrame):
    expanded = expanded.copy()

    genus_levels = sorted(expanded["genus_cat"].unique().tolist())
    if "Homo sapiens" not in genus_levels:
        raise ValueError("Expected 'Homo sapiens' among genus levels.")

    formula = (
        "amtl ~ C(genus_cat, Treatment(reference='Homo sapiens'))"
        " + age_years + prob_male + C(tooth_class)"
    )

    model = smf.logit(formula=formula, data=expanded)
    result = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": expanded["specimen_id"]},
    )

    return result


def marginal_probs_by_genus(result, expanded: pd.DataFrame) -> dict:
    probs = {}
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]:
        if genus not in expanded["genus_cat"].unique():
            continue
        df_tmp = expanded.copy()
        df_tmp["genus_cat"] = genus
        preds = result.predict(df_tmp)
        probs[genus] = float(preds.mean())
    return probs


def extract_genus_effects(robust_result) -> dict:
    """
    Extract coefficient estimates and p-values for non-human genera
    relative to Homo sapiens (reference).
    """
    effects = {}
    params = robust_result.params
    bse = robust_result.bse
    pvalues = robust_result.pvalues

    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus_cat, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term not in params.index:
            continue
        coef = float(params[term])
        se = float(bse[term])
        pval = float(pvalues[term])

        z = None
        if se > 0:
            z = coef / se
        effects[genus] = {"coef": coef, "se": se, "pval": pval, "z": z}

    return effects


def compute_likert_score(
    probs: dict,
    effects: dict,
) -> int:
    human_prob = probs.get("Homo sapiens")
    if human_prob is None:
        return 50

    nonhuman_probs = [probs[g] for g in probs.keys() if g != "Homo sapiens"]
    if not nonhuman_probs:
        return 50

    avg_nonhuman = float(np.mean(nonhuman_probs))
    diff = human_prob - avg_nonhuman

    avg_pval = np.mean([e["pval"] for e in effects.values() if "pval" in e]) if effects else 1.0

    if diff <= 0:
        base = max(0.0, 50.0 + 300.0 * diff)
        evidence_factor = 1.0 - min(1.0, (0.05 - avg_pval) / 0.05) if avg_pval < 0.05 else 1.0
        score = base * evidence_factor
        return int(round(np.clip(score, 0, 50)))

    rel_increase = diff / max(1e-6, avg_nonhuman)
    raw = 50.0 + 40.0 * np.tanh(rel_increase * 2.0)

    if avg_pval < 0.001:
        evidence_boost = 10.0
    elif avg_pval < 0.01:
        evidence_boost = 7.0
    elif avg_pval < 0.05:
        evidence_boost = 4.0
    else:
        evidence_boost = 0.0

    score = raw + evidence_boost
    return int(round(np.clip(score, 50, 100)))


def build_explanation(
    probs: dict,
    effects: dict,
    score: int,
) -> str:
    lines = []

    human_prob = probs.get("Homo sapiens")
    nonhuman_probs = {g: p for g, p in probs.items() if g != "Homo sapiens"}

    if human_prob is not None and nonhuman_probs:
        avg_nonhuman = float(np.mean(list(nonhuman_probs.values())))
        diff = human_prob - avg_nonhuman
        lines.append(
            "I modelled the probability that an individual tooth was missing "
            "using a logistic regression with a binomial likelihood."
        )
        lines.append(
            "The response was whether each tooth position showed antemortem tooth loss (1) "
            "or was present (0), constructed by expanding each row into individual teeth "
            "based on the number of observable sockets and the number of missing teeth."
        )
        lines.append(
            "Predictors included genus (Homo sapiens, Pan, Papio, Pongo), estimated age at death, "
            "a continuous sex indicator, and tooth class (anterior, posterior, premolar). "
            "I used cluster-robust standard errors at the specimen level to account for multiple "
            "teeth per individual."
        )
        lines.append(
            f"After adjusting for age, sex, and tooth class, the model-estimated mean probability "
            f"of antemortem tooth loss for a tooth in modern humans was about "
            f"{human_prob:.3f}, compared to an average of {avg_nonhuman:.3f} "
            f"across the non-human genera (Pan, Papio, Pongo). "
            f"This is a difference of roughly {diff:.3f} in absolute probability."
        )

        for genus, p in nonhuman_probs.items():
            lines.append(
                f"For {genus}, the adjusted mean probability of AMTL was approximately {p:.3f}."
            )

    for genus, eff in effects.items():
        coef = eff.get("coef")
        pval = eff.get("pval")
        if coef is None or pval is None:
            continue

        direction = "lower" if coef < 0 else "higher"
        lines.append(
            f"In the logistic model, the coefficient for {genus} relative to Homo sapiens "
            f"was {coef:.3f} on the log-odds scale, indicating {direction} odds of AMTL "
            f"for {genus} teeth compared to human teeth with the same age, sex, and tooth class."
        )
        lines.append(
            f"The corresponding p-value for this difference was {pval:.3g}, "
            "based on cluster-robust standard errors by specimen."
        )

    if score >= 50:
        yes_no = "Yes"
    else:
        yes_no = "No"

    lines.append(
        f"Putting these results together, my overall answer to the research question "
        f"('Do modern humans have higher frequencies of AMTL than non-human primates, "
        f"after accounting for age, sex, and tooth class?') is: {yes_no}."
    )
    lines.append(
        f"The Likert-scale response value of {score} (where 0 is a strong 'No' and 100 a strong 'Yes') "
        "reflects both the magnitude of the estimated human–non-human difference in AMTL frequency "
        "and the statistical evidence (p-values) from the regression model."
    )

    explanation = " ".join(lines)
    return explanation


def main() -> None:
    expanded = load_and_prepare_data("amtl.csv")
    result = fit_logistic_model(expanded)

    probs = marginal_probs_by_genus(result, expanded)
    effects = extract_genus_effects(result)
    score = compute_likert_score(probs, effects)
    explanation = build_explanation(probs, effects, score)

    conclusion = {"response": int(score), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()
