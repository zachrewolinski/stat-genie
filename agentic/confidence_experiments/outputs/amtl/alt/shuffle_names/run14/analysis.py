import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Reconstruct semantic variables based on the metadata and data patterns.
    # Column meanings (see info.json + manual inspection):
    # - sockets: tooth class ("Anterior", "Posterior", "Premolar")
    # - prob_male: specimen identifier string
    # - genus: count of missing teeth in that tooth class
    # - age: count of observable sockets in that tooth class
    # - pop: estimated age at death
    # - num_amtl: uncertainty of age at death (not used here)
    # - stdev_age: estimate of sex (probability of being male, from 0 to 1)
    # - tooth_class: genus label ("Homo sapiens", "Pan", "Papio", "Pongo")
    # - specimen: population/region label

    df = df.copy()

    # Preserve the original genus label before reusing the tooth_class name
    df["genus_label"] = df["tooth_class"]  # Homo sapiens / Pan / Papio / Pongo

    # Derived, semantically named columns
    df["tooth_class"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_est"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["prob_male_val"] = df["stdev_age"].astype(float)
    df["region"] = df["specimen"]

    # AMTL rate per specimen x tooth class
    # Guard against any potential divide-by-zero, though metadata suggests num_sockets >= 2.
    df = df[df["num_sockets"] > 0].copy()
    df["amtl_rate"] = df["num_missing"] / df["num_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Center/scale continuous covariates to make model more stable and interpretation easier.
    df["age_est_c"] = df["age_est"] - df["age_est"].mean()
    df["prob_male_c"] = df["prob_male_val"] - df["prob_male_val"].mean()

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression with grouped data:
    # response: amtl_rate with number of trials = num_sockets
    # predictors: is_human + age_est + sex + tooth_class
    formula = "amtl_rate ~ is_human + age_est_c + prob_male_c + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_human_effect(result, df: pd.DataFrame):
    # Extract coefficient for the human indicator
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pvalue = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"].to_numpy()

    # Odds ratio and CI
    odds_ratio = float(np.exp(coef))
    or_ci_low, or_ci_high = np.exp(conf_int)

    # Average marginal effect on AMTL probability:
    # Predict probabilities for each observation under human vs non-human status.
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    df_human = df.copy()
    df_human["is_human"] = 1

    # Use the fitted model to generate predictions for these counterfactuals.
    # We must use the same model formula/design, so construct a GLM with the same
    # matrix but without refitting.
    # statsmodels allows predict with modified data directly via result.predict.
    pred_nonhuman = result.predict(df_nonhuman)
    pred_human = result.predict(df_human)

    # Average predicted AMTL rates across the observed covariate distribution.
    mean_nonhuman = float(pred_nonhuman.mean())
    mean_human = float(pred_human.mean())
    diff = mean_human - mean_nonhuman

    return {
        "coef": float(coef),
        "se": float(se),
        "pvalue": float(pvalue),
        "conf_int": [float(conf_int[0]), float(conf_int[1])],
        "odds_ratio": odds_ratio,
        "or_ci": [float(or_ci_low), float(or_ci_high)],
        "mean_pred_human": mean_human,
        "mean_pred_nonhuman": mean_nonhuman,
        "mean_diff": diff,
    }


def map_effect_to_likert(human_effect: dict) -> int:
    """
    Map the strength and direction of evidence to a 0–100 Likert scale.

    0   -> strong "No" (humans clearly lower AMTL)
    50  -> no clear evidence either way
    100 -> strong "Yes" (humans clearly higher AMTL)
    """
    coef = human_effect["coef"]
    pvalue = human_effect["pvalue"]
    mean_diff = human_effect["mean_diff"]

    # Basic consistency check: direction from coefficient and mean_diff should agree.
    direction = np.sign(coef) if coef != 0 else np.sign(mean_diff)

    # If direction is negative (humans lower AMTL) and statistically clear,
    # we want a low score; if positive and clear, a high score.
    # Use p-value to adjust certainty:
    #   p < 0.001 -> very strong
    #   0.001–0.01 -> strong
    #   0.01–0.05 -> moderate
    #   0.05–0.10 -> weak / suggestive
    #   >= 0.10   -> little evidence

    if pvalue < 0.001:
        base_strength = 0.95
    elif pvalue < 0.01:
        base_strength = 0.85
    elif pvalue < 0.05:
        base_strength = 0.75
    elif pvalue < 0.10:
        base_strength = 0.60
    else:
        base_strength = 0.50

    if direction > 0:
        # Evidence that humans have higher AMTL
        score = 50 + base_strength * 50
    elif direction < 0:
        # Evidence that humans have lower AMTL
        score = 50 - base_strength * 50
    else:
        # Essentially no direction; keep near neutral
        score = 50

    # Clamp to [0, 100] and return integer
    score = int(round(min(100, max(0, score))))
    return score


def build_explanation(human_effect: dict, likert_score: int) -> str:
    direction = "higher" if human_effect["coef"] > 0 else "lower"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?\n\n"
    )

    explanation += "Data and variables:\n"
    explanation += (
        "- Each row represents a specimen-by-tooth-class combination with counts of missing teeth "
        "and observable sockets.\n"
        "- The response modeled is the proportion of missing teeth (AMTL rate) for that combination, "
        "with the binomial denominator given by the number of observable sockets.\n"
        "- Predictors include a binary indicator for modern humans vs non-human genera, "
        "estimated age at death (centered), an estimate of sex (probability of being male, centered), "
        "and categorical tooth class (anterior, posterior, premolar).\n\n"
    )

    explanation += "Modeling approach:\n"
    explanation += (
        "- I fit a binomial regression (logistic GLM) to the AMTL rate using the number of sockets as "
        "the binomial trial count, which corresponds to the binomial regression modeling described in "
        "the dataset documentation.\n"
        "- The key parameter of interest is the coefficient for the human indicator, which measures how "
        "AMTL odds differ between modern humans and non-human primates after adjusting for age, sex, "
        "and tooth class.\n\n"
    )

    explanation += "Key results for the human vs non-human contrast:\n"
    explanation += (
        f"- Estimated log-odds coefficient for humans: {human_effect['coef']:.3f} "
        f"(SE = {human_effect['se']:.3f}, p-value = {human_effect['pvalue']:.3g}).\n"
        f"- 95% confidence interval for the human coefficient: "
        f"[{human_effect['conf_int'][0]:.3f}, {human_effect['conf_int'][1]:.3f}].\n"
        f"- Corresponding odds ratio: {human_effect['odds_ratio']:.3f} "
        f"with 95% CI [{human_effect['or_ci'][0]:.3f}, {human_effect['or_ci'][1]:.3f}].\n"
        f"- Average predicted AMTL rate for non-human primates (holding the observed covariate "
        f"distribution of age, sex, and tooth class fixed): "
        f"{human_effect['mean_pred_nonhuman']:.3f}.\n"
        f"- Average predicted AMTL rate for modern humans under the same covariate distribution: "
        f"{human_effect['mean_pred_human']:.3f}.\n"
        f"- Difference in predicted AMTL frequency (humans minus non-humans): "
        f"{human_effect['mean_diff']:.3f}, indicating {direction} AMTL in humans.\n\n"
    )

    explanation += "Interpretation with respect to the research question:\n"
    if human_effect["pvalue"] < 0.05 and human_effect["coef"] > 0:
        interpretation = (
            "The positive and statistically significant coefficient for the human indicator shows that, "
            "after controlling for age, sex, and tooth class, modern humans have higher odds and higher "
            "predicted frequencies of AMTL compared to non-human primates. The confidence interval for "
            "the odds ratio lies mostly above 1, reinforcing this conclusion."
        )
        answer = (
            "Yes, the data provide evidence that modern humans exhibit higher AMTL frequencies than the "
            "non-human primate genera considered."
        )
    elif human_effect["pvalue"] < 0.05 and human_effect["coef"] < 0:
        interpretation = (
            "The negative and statistically significant coefficient for the human indicator shows that, "
            "after controlling for age, sex, and tooth class, modern humans actually have lower odds and "
            "lower predicted frequencies of AMTL compared to non-human primates. This result clearly "
            "contradicts the hypothesis that humans have higher AMTL frequencies."
        )
        answer = (
            "No, the data indicate that modern humans have lower AMTL frequencies than the non-human "
            "primates in this sample, once age, sex, and tooth class are accounted for."
        )
    elif human_effect["pvalue"] < 0.10:
        interpretation = (
            "The coefficient for the human indicator has a consistent direction but only weak statistical "
            "support (p-value between 0.05 and 0.10). This suggests a possible difference in AMTL "
            "frequencies between humans and non-human primates, but the evidence is not strong enough "
            "to be conclusive."
        )
        if human_effect["coef"] > 0:
            answer = (
                "Tentatively yes: humans may have higher AMTL frequencies than non-human primates, but "
                "the statistical evidence is only suggestive rather than definitive."
            )
        else:
            answer = (
                "Tentatively no: humans may have lower AMTL frequencies than non-human primates, but "
                "the statistical evidence is only suggestive rather than definitive."
            )
    else:
        interpretation = (
            "The coefficient for the human indicator is not statistically distinguishable from zero at "
            "conventional levels, and the confidence interval includes both meaningful positive and "
            "negative effects. After adjusting for age, sex, and tooth class, the data do not provide "
            "clear evidence that modern humans differ from non-human primates in AMTL frequencies."
        )
        answer = (
            "No clear answer: based on this dataset and model, we cannot confidently say that modern "
            "humans have higher AMTL frequencies than non-human primates once age, sex, and tooth class "
            "are controlled for."
        )

    explanation += interpretation + "\n\n"
    explanation += (
        f"Overall Likert-scale assessment (0 = strong 'No', 100 = strong 'Yes'): {likert_score}.\n"
        f"This score reflects both the direction of the estimated human effect "
        f"({'higher' if human_effect['coef'] > 0 else 'lower' if human_effect['coef'] < 0 else 'no clear difference'}) "
        "and the strength of the statistical evidence (as captured by the p-value and confidence interval). "
        "Scores near 0 or 100 correspond to strong, statistically well-supported conclusions, while scores "
        "near 50 represent weak or ambiguous evidence."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result = fit_binomial_model(df)
    human_effect = summarize_human_effect(result, df)
    likert_score = map_effect_to_likert(human_effect)
    explanation = build_explanation(human_effect, likert_score)

    conclusion = {
        "response": likert_score,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
