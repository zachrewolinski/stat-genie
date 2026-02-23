import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("amtl.csv")
OUTPUT_FILE = Path("conclusion.txt")


def run_analysis() -> None:
    df = pd.read_csv(DATA_FILE)

    # Construct variables
    df["is_human"] = (df["genus"].str.contains("Homo", case=False)).astype(int)
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Descriptive summary by genus
    genus_summary = (
        df.groupby("genus")["amtl_prop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Binomial regression: proportion of missing teeth with trial counts = sockets
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract key statistics for the human indicator
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = map(float, result.conf_int().loc["is_human"])

    # Predicted probabilities for a typical tooth (mean covariates, modal tooth class)
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    ref_tooth = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [ref_tooth, ref_tooth],
        }
    )
    pred_probs = result.predict(pred_df)
    nonhuman_prob = float(pred_probs.iloc[0])
    human_prob = float(pred_probs.iloc[1])
    diff = human_prob - nonhuman_prob

    # Map statistical evidence to a 0–100 Likert response
    if pval < 0.05:
        if coef > 0:
            # Evidence that humans have higher AMTL than non-humans
            if pval < 0.001:
                response = 95
            elif pval < 0.01:
                response = 85
            else:
                response = 75
            answer_label = "Yes"
        else:
            # Evidence that humans have equal or lower AMTL than non-humans
            if pval < 0.001:
                response = 5
            elif pval < 0.01:
                response = 15
            else:
                response = 25
            answer_label = "No"
    else:
        # No statistically significant difference after accounting for covariates
        response = 35
        answer_label = "No"

    # Build explanation text
    n_rows = len(df)
    n_specimens = df["specimen"].nunique()
    genus_means = df.groupby("genus")["amtl_prop"].mean()
    genus_parts = [f"{g}: {m:.3f}" for g, m in genus_means.items()]
    genus_text = "; ".join(genus_parts)

    p_text = f"{pval:.3g}"
    coef_text = f"{coef:.3f}"
    ci_text = f"[{ci_low:.3f}, {ci_high:.3f}]"
    diff_pct = diff * 100.0

    if pval < 0.05 and coef > 0:
        conclusion_clause = (
            "the positive and statistically significant human coefficient indicates that, "
            "after adjusting for age, estimated sex, and tooth class, modern humans have "
            "higher AMTL frequencies than the pooled non-human primate genera."
        )
    elif pval < 0.05 and coef <= 0:
        conclusion_clause = (
            "the non-positive and statistically significant human coefficient indicates that, "
            "after adjusting for age, estimated sex, and tooth class, modern humans do not have "
            "higher AMTL frequencies and instead have similar or lower AMTL than the pooled "
            "non-human primate genera."
        )
    else:
        conclusion_clause = (
            "the human coefficient is not statistically significant at the 0.05 level, so the data "
            "do not provide clear evidence that modern humans differ in AMTL frequency from the "
            "pooled non-human primate genera once age, estimated sex, and tooth class are accounted for."
        )

    explanation_sentences = [
        f"{answer_label} answer with response={response} on a 0–100 scale, where 0 means a strong 'No' and 100 means a strong 'Yes' to the question of whether modern humans have higher AMTL than non-human primates after accounting for age, sex, and tooth class.",
        f"I analysed {n_rows} tooth-class-by-specimen records from {n_specimens} unique specimens, modelling the proportion of missing teeth (num_amtl out of sockets) using a binomial logistic regression with predictors for a human-versus-non-human indicator, age at death, estimated sex (prob_male), and tooth class.",
        f"Raw mean AMTL proportions by genus were {genus_text}.",
        f"In the regression, the human indicator coefficient was {coef_text} on the log-odds scale with 95% confidence interval {ci_text} and p-value {p_text}.",
        f"At average covariate values and the modal tooth class ({ref_tooth}), the model predicts an AMTL proportion of {human_prob:.3f} for humans and {nonhuman_prob:.3f} for non-human primates, a difference of {diff_pct:.1f} percentage points.",
        f"Taken together, {conclusion_clause} This inference assumes independence of rows (no explicit random effects for repeated measurements within specimens) and a correctly specified logistic link, which are important limitations to keep in mind.",
    ]

    explanation = " ".join(explanation_sentences)

    output = {"response": int(response), "explanation": explanation}
    OUTPUT_FILE.write_text(json.dumps(output))


if __name__ == "__main__":
    run_analysis()

