import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: drop rows with missing key covariates
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Define outcome as a binomial proportion with known denominator
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Fit binomial GLM with sockets as the number of trials
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract effect of being human
    coef = float(result.params["is_human"])
    pvalue = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"].astype(float)

    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Average predicted AMTL frequency for humans vs non-humans
    base_df = df.copy()
    weights = base_df["sockets"].to_numpy()

    human_df = base_df.copy()
    human_df["is_human"] = 1
    nonhuman_df = base_df.copy()
    nonhuman_df["is_human"] = 0

    pred_human = result.predict(human_df)
    pred_nonhuman = result.predict(nonhuman_df)

    avg_pred_human = float(np.average(pred_human, weights=weights))
    avg_pred_nonhuman = float(np.average(pred_nonhuman, weights=weights))
    diff = avg_pred_human - avg_pred_nonhuman

    # Decision rule: positive, statistically significant human effect and higher predicted AMTL
    if coef > 0 and pvalue < 0.05 and avg_pred_human > avg_pred_nonhuman:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "I modeled the proportion of missing teeth (num_amtl / sockets) using a binomial GLM with "
        "logit link, treating sockets as the binomial denominator and including an indicator for "
        "modern humans (Homo sapiens), age at death, sex estimate (prob_male), and tooth class "
        "(Anterior, Posterior, Premolar) as predictors. The coefficient for the human indicator "
        f"was estimated as log-odds = {coef:.3f} (odds ratio = {odds_ratio:.2f}, 95% CI for the odds ratio "
        f"= [{or_ci_low:.2f}, {or_ci_high:.2f}], p-value = {pvalue:.4g}). After adjusting for age, sex, "
        "and tooth class, the model-predicted average AMTL frequency was "
        f"{avg_pred_human:.3f} for humans versus {avg_pred_nonhuman:.3f} for non-human primates "
        f"(difference = {diff:.3f}). Based on this analysis, the data "
        f"{'support' if response == 'Yes' else 'do not clearly support'} the claim that modern humans "
        "have higher frequencies of antemortem tooth loss than Pan, Pongo, and Papio after accounting "
        "for the specified covariates."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
