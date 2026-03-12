import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL.

    Response: proportion of missing teeth (num_amtl / sockets)
    Predictors: human vs non-human, age, prob_male, tooth_class
    Weights: number of sockets (binomial trials).
    """
    df = df.copy()

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = df["genus"].str.contains("Homo", case=False).astype(int)

    # Proportion of missing teeth; GLM Binomial with freq_weights expects a proportion
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        "prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result, df


def summarize_human_effect(result, df: pd.DataFrame):
    """Extract effect size, significance, and predicted probabilities for humans vs non-humans."""
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    beta = float(params["is_human"])
    pval = float(pvalues["is_human"])
    ci_low, ci_high = conf_int.loc["is_human"].tolist()
    or_est = float(np.exp(beta))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Average predicted AMTL proportion for humans vs. non-humans,
    # holding the distribution of covariates at their observed values.
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    pred_human = float(result.predict(df_human).mean())
    pred_nonhuman = float(result.predict(df_nonhuman).mean())
    abs_diff = pred_human - pred_nonhuman

    return {
        "beta": beta,
        "pval": pval,
        "or_est": or_est,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "pred_human": pred_human,
        "pred_nonhuman": pred_nonhuman,
        "abs_diff": abs_diff,
    }


def map_to_likert(effect):
    """
    Map the human-effect estimate and p-value to a 0–100 Likert score.

    Higher values mean stronger "Yes" that humans have higher AMTL.
    Lower values mean stronger "No".
    """
    beta = effect["beta"]
    pval = effect["pval"]
    or_est = effect["or_est"]

    # Default neutral value
    response = 50
    interpretation = ""
    yes_no = ""

    if beta > 0 and pval < 0.05:
        # Clear evidence in the hypothesized (higher human AMTL) direction
        base = 70
        if pval < 0.001:
            base = 90
        elif pval < 0.01:
            base = 80

        # Adjust for effect size (odds ratio)
        if or_est > 2.0:
            base += 5
        elif or_est < 1.2:
            base -= 5

        yes_no = "Yes"
        interpretation = "clear evidence that modern humans have higher AMTL frequencies than non-human primates after adjustment"
        response = base
    elif beta > 0 and pval >= 0.05:
        # Effect points in hypothesized direction but is not conventionally significant
        base = 55
        if pval > 0.2:
            base = 50

        yes_no = "Yes (weak evidence)"
        interpretation = "suggestive but not statistically conclusive evidence that modern humans may have higher AMTL frequencies"
        response = base
    else:
        # No or opposite effect relative to hypothesis
        base = 30
        if pval < 0.05:
            base = 10
        elif pval < 0.1:
            base = 20

        yes_no = "No"
        interpretation = "no evidence that modern humans have higher AMTL frequencies; the data do not support the hypothesized increase"
        response = base

    response_int = int(round(max(0, min(100, response))))
    return response_int, yes_no, interpretation


def build_explanation(n_rows, effect, response_int, yes_no, interpretation):
    beta = effect["beta"]
    pval = effect["pval"]
    or_est = effect["or_est"]
    or_ci_low = effect["or_ci_low"]
    or_ci_high = effect["or_ci_high"]
    pred_human = effect["pred_human"]
    pred_nonhuman = effect["pred_nonhuman"]
    abs_diff = effect["abs_diff"]

    direction = "positive" if beta > 0 else "non-positive"
    sig_phrase = (
        "statistically significant at the 0.05 level"
        if pval < 0.05
        else "not statistically significant at the 0.05 level"
    )

    explanation = (
        f"I analyzed antemortem tooth loss (AMTL) using {n_rows} tooth-class observations from modern humans "
        f"and three non-human primate genera (Pan, Papio, Pongo). For each record, I modeled the proportion of "
        f"missing teeth (num_amtl / sockets) with a binomial regression (GLM with binomial family), using the "
        f"number of observable sockets as the binomial trial count.\n\n"
        f"The predictors in the model were an indicator for modern humans (Homo sapiens vs. non-human primates), "
        f"age at death, estimated sex (prob_male), and tooth class (anterior, posterior, premolar). This structure "
        f"follows the study design and allows us to compare humans to non-human primates while accounting for age, "
        f"sex, and tooth class as potential confounders.\n\n"
        f"The estimated coefficient for the human indicator on the log-odds scale was {beta:.3f}, corresponding to "
        f"an odds ratio of {or_est:.2f} with a 95% confidence interval of [{or_ci_low:.2f}, {or_ci_high:.2f}] "
        f"(p = {pval:.3g}). On the probability scale, the model implies an average predicted AMTL proportion of "
        f"{pred_human:.3f} for modern humans versus {pred_nonhuman:.3f} for non-human primates, a difference of "
        f"{abs_diff:.3f}.\n\n"
        f"The human effect is {direction} and {sig_phrase}. Taken together, this provides {interpretation}. "
        f"Accordingly, I answer the research question as '{yes_no}' and represent the strength of this conclusion "
        f"with a Likert-scale value of {response_int} on a 0–100 scale, where 0 is a strong 'No' and 100 is a strong 'Yes'."
    )

    return explanation


def main():
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    result, df_model = fit_model(df)
    effect = summarize_human_effect(result, df_model)
    response_int, yes_no, interpretation = map_to_likert(effect)
    explanation = build_explanation(len(df_model), effect, response_int, yes_no, interpretation)

    output = {
        "response": response_int,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

