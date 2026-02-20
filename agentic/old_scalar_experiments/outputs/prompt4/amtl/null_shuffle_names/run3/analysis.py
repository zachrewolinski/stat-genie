import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Reconstruct true semantic variables from shuffled column names using info.json
    df = df.copy()
    df["tooth_class_true"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["genus_true"] = df["tooth_class"].replace({"Homo sapiens": "Homo"})
    df["num_missing"] = df["genus"].astype(float)  # number of missing teeth
    df["n_sockets"] = df["age"].astype(float)  # observable sockets
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male_true"] = df["stdev_age"].astype(float)  # 0–1 proxy for sex
    df["is_human"] = (df["genus_true"] == "Homo").astype(int)

    # Basic data cleaning: keep only rows with valid binomial counts
    mask = (df["n_sockets"] > 0) & (df["num_missing"] >= 0) & (
        df["num_missing"] <= df["n_sockets"]
    )
    df = df.loc[mask].copy()

    # Proportion of missing teeth within each tooth class for each specimen
    df["prop_missing"] = df["num_missing"] / df["n_sockets"]

    # Fit binomial regression: missing-tooth proportion ~ human + age + sex + tooth class
    model = smf.glm(
        formula="prop_missing ~ is_human + age_at_death + prob_male_true + C(tooth_class_true)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    coef_human = model.params["is_human"]
    pval_human = model.pvalues["is_human"]
    odds_ratio_human = float(np.exp(coef_human))

    # Counterfactual predicted AMTL frequencies:
    # compare humans vs non-humans while holding age, sex, and tooth class at their observed values.
    covariates = df[
        ["age_at_death", "prob_male_true", "tooth_class_true", "is_human"]
    ].copy()

    covariates_human = covariates.copy()
    covariates_human["is_human"] = 1
    covariates_nonhuman = covariates.copy()
    covariates_nonhuman["is_human"] = 0

    pred_human = float(model.predict(covariates_human).mean())
    pred_nonhuman = float(model.predict(covariates_nonhuman).mean())
    diff = pred_human - pred_nonhuman

    # Map effect size into a 0–100 Likert-style score centered at 50 (no difference).
    # Each 0.10 absolute difference in predicted AMTL prevalence moves the score by ~20 points.
    base = 50.0
    score = base + diff * 200.0
    score = int(round(max(0.0, min(100.0, score))))

    # Construct explanation text (single-line string for clean JSON).
    # Qualitative interpretation of the human effect
    if abs(diff) < 0.005:
        direction_phrase = "very similar AMTL frequencies to"
    elif diff > 0:
        direction_phrase = "higher AMTL frequencies than"
    else:
        direction_phrase = "lower AMTL frequencies than"

    if pval_human < 0.05:
        if diff > 0:
            conclusion_phrase = (
                "This provides statistical evidence that modern humans have higher AMTL "
                "frequencies than the non-human primates after accounting for age, sex, "
                "and tooth class."
            )
        elif diff < 0:
            conclusion_phrase = (
                "This provides statistical evidence that modern humans do not have higher, "
                "and may in fact have lower, AMTL frequencies than the non-human primates "
                "after accounting for age, sex, and tooth class."
            )
        else:
            conclusion_phrase = (
                "This provides statistical evidence that AMTL frequencies for modern humans "
                "and the non-human primates are essentially the same after accounting for "
                "age, sex, and tooth class."
            )
    else:
        if abs(diff) < 0.005:
            conclusion_phrase = (
                "These results do not provide clear statistical evidence that modern humans "
                "differ meaningfully in AMTL frequency from the non-human primates once age, "
                "sex, and tooth class are accounted for."
            )
        elif diff > 0:
            conclusion_phrase = (
                "Although the point estimate suggests slightly higher AMTL frequencies for "
                "modern humans, the large p-value indicates no clear statistical evidence "
                "that humans truly have higher AMTL frequencies than the non-human primates "
                "after adjusting for age, sex, and tooth class."
            )
        else:
            conclusion_phrase = (
                "Although the point estimate suggests slightly lower AMTL frequencies for "
                "modern humans, the large p-value indicates no clear statistical evidence "
                "that humans truly differ in AMTL frequency from the non-human primates "
                "after adjusting for age, sex, and tooth class."
            )

    explanation = (
        "I modeled the proportion of antemortem tooth loss (AMTL) in each tooth class "
        "as a binomial outcome (number of missing teeth divided by observable sockets) "
        "using a logistic regression with a human-versus-nonhuman indicator, age at death, "
        "a sex proxy, and tooth class (anterior, posterior, premolar) as predictors. "
        f"The coefficient for the human indicator on the log-odds scale is {coef_human:.3f}, "
        f"corresponding to an odds ratio of {odds_ratio_human:.2f} (p = {pval_human:.3g}). "
        "When I compute counterfactual predicted AMTL frequencies while holding age, sex, "
        "and tooth class fixed, the mean predicted proportion of teeth lost is "
        f"{pred_human:.3f} for modern humans and {pred_nonhuman:.3f} for the non-human primates, "
        f"a difference of {diff:.3f} on the raw probability scale. "
        f"Overall, this pattern indicates that modern humans have {direction_phrase} the "
        "non-human primates in this dataset, conditional on age, sex, and tooth class. "
        + conclusion_phrase
    )

    conclusion = {"response": score, "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
