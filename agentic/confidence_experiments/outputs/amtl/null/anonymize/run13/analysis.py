import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_likert_response(coef: float, pval: float, odds_ratio: float) -> tuple[int, str]:
    """
    Map the evidence about the human effect to a 0–100 Likert score.

    Returns (score, label) where label is "yes" or "no" for narrative use.
    """
    alpha = 0.05

    # Not statistically significant: treat as "no" / lack of evidence.
    if not np.isfinite(pval) or pval >= alpha:
        # Start from moderate "no" and adjust by p-value strength.
        if pval >= 0.5:
            base = 15
        elif pval >= 0.2:
            base = 25
        elif pval >= 0.1:
            base = 35
        else:
            base = 45

        score = int(round(np.clip(base, 0, 100)))
        return score, "no"

    # Statistically significant effect: direction given by coefficient sign.
    if coef > 0:
        # Humans have higher AMTL frequency.
        if pval < 0.001:
            base = 92
        elif pval < 0.01:
            base = 88
        else:
            base = 80

        # Adjust by effect size (odds ratio).
        if odds_ratio < 1.1:
            base -= 10
        elif odds_ratio < 1.3:
            base -= 5
        elif odds_ratio > 2.0:
            base += 5
        elif odds_ratio > 3.0:
            base += 8

        score = int(round(np.clip(base, 0, 100)))
        return score, "yes"

    # Significant but negative coefficient: humans have lower AMTL frequency.
    if pval < 0.001:
        base = 5
    elif pval < 0.01:
        base = 10
    else:
        base = 20

    # Larger deviation of odds_ratio from 1 implies stronger "no".
    if odds_ratio < 0.7:
        base -= 5
    elif odds_ratio > 0.9:
        base += 5

    score = int(round(np.clip(base, 0, 100)))
    return score, "no"


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic preparation
    df = df.copy()
    df["missing"] = df["feature3"]
    df["total"] = df["feature4"]

    # Exclude any rows with non-positive socket counts.
    df = df[df["total"] > 0].copy()

    # Key predictors
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)
    df["age"] = df["feature5"]
    df["sex_est"] = df["feature7"]

    # Tooth class as categorical
    df["tooth_class"] = df["feature1"].astype("category")

    # Expand to socket-level data to fit a standard logistic regression:
    # each observable socket becomes a binary trial (AMTL present vs. not).
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        total = int(row["total"])
        missing = int(row["missing"])
        if total <= 0 or missing < 0 or missing > total:
            continue
        for i in range(total):
            expanded_rows.append(
                {
                    "amtl": 1 if i < missing else 0,
                    "is_human": row["is_human"],
                    "age": row["age"],
                    "sex_est": row["sex_est"],
                    "tooth_class": row["tooth_class"],
                }
            )

    df_long = pd.DataFrame(expanded_rows)

    # Binomial regression: logit(p(AMTL)) = is_human + age + sex + tooth_class
    model = smf.glm(
        "amtl ~ is_human + age + sex_est + C(tooth_class)",
        data=df_long,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # 95% CI on odds ratio
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    score, label = compute_likert_response(coef, pval, odds_ratio)

    n_rows = int(df.shape[0])
    n_humans = int(df[df["is_human"] == 1].shape[0])
    n_non_humans = n_rows - n_humans
    n_sockets = int(df_long.shape[0])

    if label == "yes":
        human_sentence = (
            "Because the human indicator is positive and statistically significant, "
            "the model supports the conclusion that modern humans have higher AMTL "
            "frequencies than the non-human primates examined, after accounting for "
            "age, sex, and tooth class.\n\n"
        )
    else:
        human_sentence = (
            "Because the human indicator is not statistically significantly greater "
            "(or is significantly lower), the model does not support the claim that "
            "modern humans have higher AMTL frequencies than the non-human primates "
            "once age, sex, and tooth class are taken into account.\n\n"
        )

    explanation_parts = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "controlling for age, sex, and tooth class?\n\n",
        f"I analyzed the dataset of {n_rows} specimen–tooth-class observations (expanded to {n_sockets} individual tooth sockets) using a binomial "
        "logistic regression model of the proportion of missing teeth (missing sockets / observable sockets). "
        "The model included an indicator for modern humans versus non-human primates, estimated age at death, "
        "estimated sex, and tooth class (anterior, posterior, premolar) as predictors. Each row was weighted by "
        "the number of observable sockets for that tooth class.\n\n",
        f"The coefficient for the human indicator (Homo sapiens vs. non-human primates) on the log-odds scale was "
        f"{coef:.3f}, corresponding to an odds ratio of {odds_ratio:.2f} "
        f"(95% CI [{or_ci_low:.2f}, {or_ci_high:.2f}], p-value = {pval:.3g}). "
        f"There were {n_humans} human observations and {n_non_humans} non-human primate observations.\n\n",
        human_sentence,
        f"I translated this evidence into a 0–100 Likert-style confidence score, where 0 represents a strong 'No' answer "
        f"and 100 represents a strong 'Yes' answer to the research question. The resulting score of {score} reflects a "
        f"{'confident affirmative' if label == 'yes' else 'confident negative or lack-of-evidence'} conclusion given the "
        "estimated effect size and its statistical significance.",
    ]

    explanation = "".join(explanation_parts)

    output = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
