import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL proportion with weights equal to the number of sockets.
    """
    # Keep only rows with valid socket counts and where missing teeth do not exceed sockets
    df = df.loc[(df["sockets"] > 0) & (df["num_amtl"] <= df["sockets"])].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categorical encoding for tooth_class
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Proportion of missing teeth
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + tooth_class",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return df, result


def summarize_effect(df: pd.DataFrame, result):
    """
    Extract key statistics for the human vs non-human effect.
    """
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    se = float(result.bse["is_human"])
    zval = float(result.tvalues["is_human"])
    or_human = float(np.exp(coef))
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted probabilities for a typical individual (mean age/sex, reference tooth class)
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    ref_tooth = df["tooth_class"].cat.categories[0]

    pred_df = pd.DataFrame(
        {
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [ref_tooth, ref_tooth],
            "is_human": [0, 1],
        }
    )
    pred = result.get_prediction(pred_df).predicted_mean
    # `pred` may be a NumPy array; index it positionally.
    p_nonhuman = float(pred[0])
    p_human = float(pred[1])

    diff = p_human - p_nonhuman
    ratio = p_human / p_nonhuman if p_nonhuman > 0 else np.inf

    return {
        "coef": coef,
        "pval": pval,
        "se": se,
        "zval": zval,
        "or_human": or_human,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "p_nonhuman": p_nonhuman,
        "p_human": p_human,
        "diff": diff,
        "ratio": ratio,
        "mean_age": mean_age,
        "mean_prob_male": mean_prob_male,
        "ref_tooth": str(ref_tooth),
    }


def compute_likert_from_effect(coef: float, zval: float, or_human: float) -> int:
    """
    Map the sign, significance, and size of the human effect onto a 0-100 Likert scale.

    0   = strong "No, humans do NOT have higher AMTL"
    50  = no clear evidence either way
    100 = strong "Yes, humans DO have higher AMTL"
    """
    # If coefficient is exactly zero (unlikely), return neutral.
    if coef == 0.0:
        return 50

    # Confidence component based on |z|; |z|≈2 is conventional 0.05 threshold,
    # |z|≈3 is quite strong evidence. Cap at 1.
    conf_component = min(1.0, abs(zval) / 3.0)

    # Effect-size component based on distance of OR from 1, capped at OR≈3.
    effect_component = min(1.0, abs(np.log(or_human)) / np.log(3.0))

    combined_strength = 0.5 * conf_component + 0.5 * effect_component

    if coef > 0:
        # Evidence that humans have higher AMTL
        score = 50 + round(50 * combined_strength)
    else:
        # Evidence that humans have equal or lower AMTL
        score = 50 - round(50 * combined_strength)

    # Ensure integer in [0, 100]
    score = max(0, min(100, int(score)))
    return score


def build_explanation(df: pd.DataFrame, stats: dict, response: int) -> str:
    """
    Construct a human-readable explanation of the analysis and findings.
    """
    n_total = int(len(df))
    n_human = int((df["is_human"] == 1).sum())
    n_nonhuman = n_total - n_human

    direction = "higher" if stats["coef"] > 0 else "lower or similar"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class? "
        "I analyzed the provided AMTL dataset ({} genus-tooth-class observations: {} human and {} non-human) "
        "using binomial regression (GLM with logit link), modeling the proportion of missing teeth "
        "per specimen and tooth class (num_amtl / sockets) with sockets as binomial trial counts. "
        "Predictors included an indicator for modern humans versus non-human primates, estimated age at death, "
        "probability of being male (prob_male), and tooth_class (categorical). "
        "This approach directly tests whether humans have different AMTL frequencies once age, sex, and tooth class "
        "are statistically controlled. "
        "The estimated coefficient for the human indicator is {:.3f} (SE {:.3f}, z = {:.2f}, p = {:.3g}), "
        "corresponding to an odds ratio of {:.2f} for AMTL in humans relative to non-human primates "
        "with a 95% confidence interval from {:.2f} to {:.2f}. "
        "For a typical individual at the sample-average age ({:.1f} years) and sex probability (prob_male ≈ {:.2f}) "
        "and for the reference tooth class ({!s}), the model predicts an AMTL proportion of {:.3%} for non-human primates "
        "and {:.3%} for humans, a difference of {:.3%} (human/non-human ratio ≈ {:.2f}). "
        "These results indicate that, after adjusting for age, sex, and tooth class, modern humans have {} AMTL frequencies "
        "than the non-human primates in this sample. "
        "Translating the strength of this evidence (combining both statistical significance and effect size) "
        "onto a 0–100 Likert-style scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the research question, "
        "I assign a score of {}. "
        "Values above 50 represent increasing confidence that humans have higher AMTL frequencies, "
        "while values below 50 would represent evidence that they do not."
    ).format(
        n_total,
        n_human,
        n_nonhuman,
        stats["coef"],
        stats["se"],
        stats["zval"],
        stats["pval"],
        stats["or_human"],
        stats["or_ci_low"],
        stats["or_ci_high"],
        stats["mean_age"],
        stats["mean_prob_male"],
        stats["ref_tooth"],
        stats["p_nonhuman"],
        stats["p_human"],
        stats["diff"],
        stats["ratio"],
        direction,
        response,
    )

    return explanation


def main():
    # Load data
    df = pd.read_csv("amtl.csv")

    # Fit model
    df_model, result = fit_model(df)

    # Print a concise summary to stdout for transparency (does not affect conclusion.txt)
    print(result.summary())

    # Extract effect summary
    stats = summarize_effect(df_model, result)

    # Compute Likert-scale response
    response = compute_likert_from_effect(
        coef=stats["coef"],
        zval=stats["zval"],
        or_human=stats["or_human"],
    )

    # Build explanation text
    explanation = build_explanation(df_model, stats, response)

    # Write JSON conclusion to file (no extra text or lines)
    output = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
