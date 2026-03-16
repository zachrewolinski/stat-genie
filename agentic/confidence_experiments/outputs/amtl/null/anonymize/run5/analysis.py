import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    """
    Fit binomial logistic regression for AMTL proportion with
    human vs non-human primate indicator, age, sex, and tooth class.
    Uses cluster-robust standard errors by specimen.
    """
    df = df.copy()

    # Basic cleaning / derived variables
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex_est",
            "feature8": "genus",
        }
    )

    # Keep only rows with positive socket counts
    df = df[df["n_sockets"] > 0].copy()

    # Indicator for modern humans (Homo sapiens) vs non-human primates
    df["human"] = df["genus"].astype(str).str.contains("Homo").astype(int)

    # Proportion of missing teeth for the given class
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Drop rows with missing key values just in case
    df = df.dropna(subset=["prop_missing", "age", "sex_est", "tooth_class", "human", "specimen_id"])

    # Ensure types
    df["tooth_class"] = df["tooth_class"].astype("category")
    df["specimen_id"] = df["specimen_id"].astype("category")

    formula = "prop_missing ~ human + age + sex_est + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )

    # Cluster-robust SEs at the specimen level
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen_id"]})

    return df, result


def summarize_human_effect(df: pd.DataFrame, result: sm.GLM):
    """Extract key quantities for the human vs non-human effect."""
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    coef_human = float(params["human"])
    pval_human = float(pvalues["human"])
    ci_low, ci_high = [float(x) for x in conf_int.loc["human"]]

    or_human = float(np.exp(coef_human))
    or_ci_low, or_ci_high = [float(np.exp(ci_low)), float(np.exp(ci_high))]

    # Average predicted probabilities under counterfactual human/non-human status
    design = df.copy()
    design_h0 = design.copy()
    design_h0["human"] = 0
    design_h1 = design.copy()
    design_h1["human"] = 1

    pred_h0 = result.predict(design_h0)
    pred_h1 = result.predict(design_h1)

    weights = df["n_sockets"].to_numpy()
    mean_prob_h0 = float(np.average(pred_h0, weights=weights))
    mean_prob_h1 = float(np.average(pred_h1, weights=weights))
    risk_diff = float(mean_prob_h1 - mean_prob_h0)

    return {
        "coef_human": coef_human,
        "pval_human": pval_human,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "or_human": or_human,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "mean_prob_h0": mean_prob_h0,
        "mean_prob_h1": mean_prob_h1,
        "risk_diff": risk_diff,
    }


def map_to_likert(effect_summary):
    """
    Map the human effect (sign, significance, and magnitude)
    to a 0–100 Likert score where higher means stronger "Yes"
    that humans have higher AMTL than non-human primates.
    """
    coef = effect_summary["coef_human"]
    pval = effect_summary["pval_human"]
    or_human = effect_summary["or_human"]
    risk_diff = effect_summary["risk_diff"]

    # Default: "No" with low confidence
    response = 20

    if pval < 0.05 and coef > 0:
        # Statistically significant evidence that humans have higher AMTL
        base = 70
        # Strengthen for more extreme significance
        if pval < 0.001:
            base += 10
        elif pval < 0.01:
            base += 5

        # Strengthen for larger effect sizes
        if or_human >= 2.0 or risk_diff >= 0.10:
            base += 10
        elif or_human >= 1.5 or risk_diff >= 0.05:
            base += 5

        response = base
    elif pval < 0.05 and coef < 0:
        # Statistically significant evidence that humans have LOWER AMTL
        base = 30
        if pval < 0.001:
            base -= 10
        elif pval < 0.01:
            base -= 5

        if or_human <= 0.5 or risk_diff <= -0.10:
            base -= 10
        elif or_human <= 0.67 or risk_diff <= -0.05:
            base -= 5

        response = base
    else:
        # Non-significant results: evidence insufficient for a clear yes/no
        if pval >= 0.5:
            response = 20
        elif pval >= 0.1:
            response = 30
        else:
            # Marginal (0.05 <= p < 0.1)
            response = 40

    # Ensure bounds and integer
    response = int(round(max(0, min(100, response))))

    # Yes/No textual label driven by whether evidence supports humans having higher AMTL
    if pval < 0.05 and coef > 0:
        answer = "Yes"
    else:
        answer = "No"

    return response, answer


def build_explanation(df: pd.DataFrame, effect_summary, response: int, answer: str) -> str:
    """Construct a human-readable explanation of the analysis and findings."""
    n_obs = len(df)
    n_specimens = df["specimen_id"].nunique()
    genus_counts = df["genus"].value_counts().to_dict()

    coef = effect_summary["coef_human"]
    pval = effect_summary["pval_human"]
    or_human = effect_summary["or_human"]
    or_ci_low = effect_summary["or_ci_low"]
    or_ci_high = effect_summary["or_ci_high"]
    mean_prob_h0 = effect_summary["mean_prob_h0"]
    mean_prob_h1 = effect_summary["mean_prob_h1"]
    risk_diff = effect_summary["risk_diff"]

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Papio, Pongo) after "
        "accounting for age, sex, and tooth class?\n\n"
        f"Data and design: The dataset contains {n_obs} observations, each representing a combination "
        "of specimen and tooth class, with counts of missing teeth (feature3) out of observable "
        "sockets (feature4). There are "
        f"{n_specimens} unique specimens, drawn from genera {genus_counts}. I modeled the proportion "
        "of missing teeth in each row as a binomial outcome with the number of observable sockets as "
        "the trial count. Predictors included a binary indicator for modern humans versus non-human "
        "primates, estimated age at death, an estimated sex score, and tooth class "
        "(anterior/posterior/premolar). To respect the aggregation by specimen, I used cluster-robust "
        "standard errors with clusters at the specimen level.\n\n"
        "Modeling approach: I fit a binomial logistic regression using statsmodels GLM with a logit "
        "link, modeling the AMTL proportion (missing teeth / observable sockets) as a function of "
        "the human indicator, age, sex estimate, and categorical tooth class. The model was fit with "
        "binomial weights equal to the number of observable sockets, so rows with more teeth "
        "contribute proportionally more information.\n\n"
        "Key human–non-human contrast: The regression coefficient for the modern human indicator "
        f"was {coef:.3f}, corresponding to an odds ratio of {or_human:.2f} "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}), with p-value {pval:.4g}. Using the fitted model, "
        "I computed model-implied mean AMTL probabilities by setting the human indicator to 0 or 1 "
        "for all observations while keeping age, sex, and tooth class fixed at their observed values. "
        f"This yields an average predicted AMTL probability of {mean_prob_h1:.3f} for humans versus "
        f"{mean_prob_h0:.3f} for non-human primates, a risk difference of {risk_diff:.3f}.\n\n"
        "Interpretation and conclusion: "
    )

    if pval < 0.05 and coef > 0:
        interpretation = (
            "After adjusting for age, sex, and tooth class, there is statistically significant evidence "
            "that modern humans have higher AMTL frequencies than the non-human primates in this sample. "
        )
    elif pval < 0.05 and coef < 0:
        interpretation = (
            "After adjusting for age, sex, and tooth class, there is statistically significant evidence "
            "that modern humans have lower AMTL frequencies than the non-human primates in this sample. "
        )
    else:
        interpretation = (
            "After adjusting for age, sex, and tooth class, the estimated human effect is not statistically "
            "significant at conventional thresholds, so the data do not provide strong evidence that modern "
            "humans differ from non-human primates in AMTL frequency in this sample. "
        )

    explanation += interpretation
    explanation += (
        f"Based on the sign, magnitude, and statistical significance of the human coefficient, as well as "
        f"the estimated difference in predicted AMTL probabilities, I answer the research question with "
        f"'{answer}' and place this conclusion at {response} on a 0–100 scale, where higher values indicate "
        "stronger evidence that modern humans have higher AMTL frequencies than the non-human primates."
    )

    return explanation


def main():
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    df_modeled, result = fit_model(df)
    effect_summary = summarize_human_effect(df_modeled, result)
    response, answer = map_to_likert(effect_summary)
    explanation = build_explanation(df_modeled, effect_summary, response, answer)

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON object to conclusion.txt with no extra text
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

