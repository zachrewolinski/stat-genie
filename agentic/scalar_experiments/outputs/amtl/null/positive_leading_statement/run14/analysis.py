import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata / research question
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "amtl.csv"
    df = pd.read_csv(data_path)

    # Basic derived variable: proportion of missing teeth in this tooth class
    df = df.copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Descriptive summary by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_amtl=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            n_rows=("genus", "size"),
        )
        .reset_index()
    )
    genus_summary["prop_amtl"] = (
        genus_summary["total_amtl"] / genus_summary["total_sockets"]
    )

    # Fit binomial regression: AMTL proportion ~ genus + age + sex + tooth class
    # Using Binomial family with var_weights = number of sockets (trials)
    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    ).fit()

    # Standardized predictions for each genus:
    # For each record, substitute genus while keeping age, sex, tooth_class the same,
    # then average predicted AMTL proportion across all records.
    genera = sorted(df["genus"].unique())
    human_label = "Homo sapiens"
    if human_label not in genera:
        raise ValueError("Expected 'Homo sapiens' in genus column.")

    def predict_for_genus(genus_name: str) -> float:
        tmp = df.copy()
        tmp["genus"] = genus_name
        preds = model.predict(tmp)
        return float(preds.mean())

    predicted_means = {g: predict_for_genus(g) for g in genera}

    # Collect contrasts: each non-human genus vs modern humans
    contrast_info = {}
    for g in genera:
        if g == human_label:
            continue
        param_name = f"C(genus)[T.{g}]"
        if param_name not in model.params:
            # Should not happen, but guard defensively.
            continue

        coef = float(model.params[param_name])
        se = float(model.bse[param_name])
        p_value = float(model.pvalues[param_name])

        # In treatment coding with humans as baseline, this coefficient is:
        # logit(p_g) - logit(p_humans). Negative value => humans have higher AMTL.
        # Convert to odds ratio for humans vs this genus.
        odds_ratio_human_vs_g = float(np.exp(-coef))

        human_pred = predicted_means[human_label]
        genus_pred = predicted_means[g]
        diff = human_pred - genus_pred

        direction = "higher" if diff > 0 else "lower" if diff < 0 else "similar"
        significant = p_value < 0.05

        contrast_info[g] = {
            "genus": g,
            "coef_genus_minus_humans": coef,
            "se": se,
            "p_value": p_value,
            "odds_ratio_human_vs_genus": odds_ratio_human_vs_g,
            "human_pred": human_pred,
            "genus_pred": genus_pred,
            "difference": diff,
            "direction": direction,
            "significant": significant,
        }

    # Map evidence to a 0–100 Likert-style score (0 = strong No, 100 = strong Yes)
    score = 50.0
    for g, info_g in contrast_info.items():
        diff = info_g["difference"]
        p = info_g["p_value"]

        if diff > 0:  # humans predicted to have higher AMTL
            if p < 1e-3:
                score += 15
            elif p < 1e-2:
                score += 12
            elif p < 5e-2:
                score += 8
            elif p < 1e-1:
                score += 4
            else:
                score += 2
        elif diff < 0:  # humans predicted to have lower AMTL
            if p < 1e-3:
                score -= 15
            elif p < 1e-2:
                score -= 12
            elif p < 5e-2:
                score -= 8
            elif p < 1e-1:
                score -= 4
            else:
                score -= 2
        # If diff is extremely small (~0), we leave score unchanged.

    # Clamp and round score to integer in [0, 100]
    score_int = int(round(max(0.0, min(100.0, score))))

    if score_int >= 60:
        qualitative_answer = "Yes"
    elif score_int <= 40:
        qualitative_answer = "No"
    else:
        qualitative_answer = "Inconclusive"

    # Build human-readable explanation string
    lines = []
    if research_question:
        lines.append(
            f"Research question: {research_question.strip()}"
        )

    lines.append(
        "I analyzed the 'amtl.csv' dataset (1450 tooth-class-by-specimen records) "
        "using binomial regression to compare antemortem tooth loss (AMTL) "
        "between modern humans (Homo sapiens) and three non-human primate genera "
        "(Pan, Papio, Pongo), while controlling for age at death, sex (prob_male), "
        "and tooth class (anterior, premolar, posterior)."
    )

    # Raw descriptive differences by genus
    lines.append(
        "First, I computed simple descriptive AMTL frequencies by genus, pooling across "
        "age, sex, and tooth class:"
    )
    for _, row in genus_summary.sort_values("genus").iterrows():
        genus_name = row["genus"]
        prop_pct = 100.0 * row["prop_amtl"]
        lines.append(
            f"- {genus_name}: {row['total_amtl']} missing teeth out of "
            f"{row['total_sockets']} sockets "
            f"({prop_pct:.1f}% of sockets showing AMTL)."
        )

    # Model-based results
    lines.append(
        "Next, I fit a binomial generalized linear model with a logit link, modeling the "
        "proportion of missing teeth in each record (num_amtl / sockets) as a function "
        "of genus (treatment-coded with Homo sapiens as the reference), age, prob_male, "
        "and tooth_class, using the number of sockets as binomial trial weights."
    )

    human_pred_pct = 100.0 * predicted_means[human_label]
    lines.append(
        "Using this model, I calculated standardized predicted AMTL frequencies by genus, "
        "holding age, sex, and tooth class at their observed values and varying only genus. "
        f"For modern humans, the average predicted AMTL frequency is {human_pred_pct:.1f}%."
    )

    for g, info_g in contrast_info.items():
        genus_pred_pct = 100.0 * info_g["genus_pred"]
        direction = info_g["direction"]
        p_val = info_g["p_value"]
        or_ratio = info_g["odds_ratio_human_vs_genus"]

        if direction == "higher":
            dir_phrase = "higher"
        elif direction == "lower":
            dir_phrase = "lower"
        else:
            dir_phrase = "similar"

        if p_val < 0.001:
            sig_phrase = "highly statistically significant (p < 0.001)"
        elif p_val < 0.01:
            sig_phrase = "statistically significant (p < 0.01)"
        elif p_val < 0.05:
            sig_phrase = "statistically significant (p < 0.05)"
        else:
            sig_phrase = f"not statistically significant (p = {p_val:.3g})"

        lines.append(
            f"- Compared to {g}, humans have {dir_phrase} predicted AMTL, with "
            f"standardized predicted AMTL of {human_pred_pct:.1f}% for humans vs "
            f"{genus_pred_pct:.1f}% for {g}. The genus coefficient for {g} relative to "
            f"humans implies an odds ratio of approximately {or_ratio:.2f} for AMTL in "
            f"humans vs {g} ({sig_phrase})."
        )

    lines.append(
        "These model-based results incorporate age, sex, and tooth-class effects, so the "
        "genus comparisons reflect differences in AMTL frequencies after accounting for "
        "these covariates."
    )

    lines.append(
        f"Based on the direction and statistical significance of the genus coefficients, "
        f"I summarize the evidence as: overall answer = '{qualitative_answer}' to the "
        "question of whether modern humans have higher AMTL frequencies than non-human "
        f"primates, with a strength rating of {score_int} on a 0–100 scale "
        "(0 = strong 'No', 100 = strong 'Yes')."
    )

    explanation = "\n".join(lines)

    conclusion = {
        "response": score_int,
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)

    # Also print a brief summary for interactive inspection (not written to file).
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()

