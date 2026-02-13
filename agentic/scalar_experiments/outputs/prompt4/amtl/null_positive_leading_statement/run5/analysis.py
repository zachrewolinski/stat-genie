import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic sanity filter: keep rows with positive sockets and non-missing covariates.
    df = df[
        (df["sockets"] > 0)
        & df["num_amtl"].ge(0)
        & df["age"].notna()
        & df["prob_male"].notna()
        & df["tooth_class"].notna()
        & df["genus"].notna()
    ].copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth for descriptive summaries and binomial GLM.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Descriptive AMTL frequencies by genus (pooled over tooth classes).
    genus_summary = (
        df.groupby("genus")
        .agg(total_amtl=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(prop_amtl=lambda g: g["total_amtl"] / g["total_sockets"])
    )

    # Binomial regression: probability of AMTL controlling for age, sex, and tooth class.
    # Use sockets as frequency weights so each row represents multiple trials.
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + tooth_class",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = float(model.params["is_human"])
    pval = float(model.pvalues["is_human"])
    or_point = float(np.exp(coef))
    ci_logit = model.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(ci_logit[0]))
    or_ci_high = float(np.exp(ci_logit[1]))

    # Predicted AMTL probability for a typical specimen (mean age/sex, common tooth class).
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    common_tooth_class = df["tooth_class"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )
    preds = model.predict(pred_df)
    nonhuman_pred = float(preds.iloc[0])
    human_pred = float(preds.iloc[1])
    diff_pred = human_pred - nonhuman_pred

    # Map evidence strength to a 0-100 Likert-style response.
    if coef > 0 and pval < 0.001:
        response = 95
        qualitative = "strong"
        answer = "Yes"
    elif coef > 0 and pval < 0.01:
        response = 85
        qualitative = "moderate-to-strong"
        answer = "Yes"
    elif coef > 0 and pval < 0.05:
        response = 75
        qualitative = "moderate"
        answer = "Yes"
    elif coef > 0:
        response = 60
        qualitative = "weak"
        answer = "Yes (but the evidence is weak)"
    elif coef < 0 and pval < 0.001:
        response = 5
        qualitative = "strong"
        answer = "No"
    elif coef < 0 and pval < 0.01:
        response = 15
        qualitative = "moderate-to-strong"
        answer = "No"
    elif coef < 0 and pval < 0.05:
        response = 25
        qualitative = "moderate"
        answer = "No"
    else:
        response = 50
        qualitative = "inconclusive"
        answer = "Uncertain"

    # Pull simple genus-level proportions for context in the explanation.
    genus_props = {genus: float(prop) for genus, prop in genus_summary["prop_amtl"].to_dict().items()}
    human_prop = genus_props.get("Homo sapiens", float("nan"))
    nonhuman_genus = {g: p for g, p in genus_props.items() if g != "Homo sapiens"}

    # Build a concise, human-readable explanation.
    nonhuman_parts = [
        f"{g}: {p * 100:.1f}% of sockets missing" for g, p in sorted(nonhuman_genus.items())
    ]
    nonhuman_text = "; ".join(nonhuman_parts)

    explanation = (
        "I modeled the proportion of missing teeth (num_amtl / sockets) using a binomial "
        "regression with a logit link, including predictors for human vs non-human primate "
        "(is_human), age at death, sex (prob_male), and tooth class. "
        f"In this model the coefficient for modern humans on the log-odds scale was {coef:.3f}, "
        f"corresponding to an odds ratio of {or_point:.2f} "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, p = {pval:.4g}). "
        f"At mean age, mean sex estimate, and for the most common tooth class ({common_tooth_class}), "
        f"the predicted AMTL probability was {nonhuman_pred * 100:.1f}% for non-human primates "
        f"and {human_pred * 100:.1f}% for humans (a difference of {diff_pred * 100:.1f} percentage points). "
        "Descriptively, when pooling over tooth classes, the proportion of missing teeth was "
        f"{human_prop * 100:.1f}% of sockets for Homo sapiens and "
        f"{nonhuman_text} for the non-human genera. "
        f"Overall, the evidence for humans having higher AMTL frequencies than non-human primates, "
        f"after accounting for age, sex, and tooth class, is {qualitative}; therefore my answer to the "
        f"research question is: {answer}."
    )

    output = {"response": int(response), "explanation": explanation}
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
