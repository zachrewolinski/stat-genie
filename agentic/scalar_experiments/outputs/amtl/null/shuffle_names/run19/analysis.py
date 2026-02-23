import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_likert_score(coef_human: float, pval_human: float) -> int:
    """
    Map the Homo sapiens coefficient and p-value to a 0-100 Likert score.

    Scores >= 50 correspond to a "Yes" answer (evidence that humans have
    higher AMTL frequencies), and scores < 50 correspond to a "No" answer.
    Only statistically significant positive effects (p < 0.05) are treated as
    evidence for a "Yes".
    """
    # Statistically significant higher AMTL in humans
    if coef_human > 0 and pval_human < 0.05:
        if pval_human < 0.001:
            return 95
        if pval_human < 0.01:
            return 85
        return 70

    # No statistically significant evidence that humans have higher AMTL
    # (these are all treated as "No" answers with varying strength).
    if coef_human > 0:
        # Trend toward higher AMTL in humans but not significant
        if pval_human < 0.1:
            return 40
        return 30

    # Coefficient is zero or negative (humans same or lower AMTL)
    if pval_human < 0.05:
        return 10
    if pval_human < 0.1:
        return 20
    return 25


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # The column names in this shuffled dataset are not semantically aligned
    # with their original names, so we rely on the descriptions in info.json
    # and inspection of example rows:
    #
    # - "genus": number of missing teeth of the given class (AMTL count)
    # - "age": number of observable sockets that could be scored
    # - "pop": estimated age at death
    # - "stdev_age": probability specimen is male (0-1)
    # - "sockets": tooth class within the mouth (Anterior/Posterior/Premolar)
    # - "tooth_class": specimen genus (Homo sapiens, Pan, Papio, Pongo)

    df = df.copy()
    df["missing"] = df["genus"].astype(float)
    df["sockets_count"] = df["age"].astype(float)

    # Keep only rows with valid counts
    valid = (
        df["sockets_count"] > 0
    ) & (df["missing"] >= 0) & (df["missing"] <= df["sockets_count"])
    df = df.loc[valid].copy()

    # Response as proportion missing with binomial denominator
    df["prop_missing"] = df["missing"] / df["sockets_count"]

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["tooth_class"] == "Homo sapiens").astype(int)

    # Covariates: age at death, sex, tooth class
    df["age_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)
    df["sex_male"] = (df["prob_male"] >= 0.5).astype(int)
    df["tooth_type"] = df["sockets"].astype(str)

    # Fit binomial regression for AMTL proportion with logit link
    model = smf.glm(
        formula="prop_missing ~ is_human + age_death + sex_male + C(tooth_type)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets_count"],
    )
    result = model.fit()

    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))

    score = compute_likert_score(coef_human, pval_human)
    yes_no = "Yes" if score >= 50 else "No"

    # Build explanation string
    direction = "higher" if coef_human > 0 else "lower"
    sig_text = (
        "statistically significant at the 0.05 level"
        if pval_human < 0.05
        else "not statistically significant at the 0.05 level"
    )

    explanation_parts = [
        "I modeled the frequency of antemortem tooth loss (AMTL) using a binomial ",
        "logistic regression. For each specimen and tooth class, I treated the ",
        '"genus" column as the count of missing teeth and the "age" column as the ',
        "number of observable sockets, modeling the proportion missing with a ",
        "binomial logit link and the socket count as the denominator. ",
        "The main predictor of interest was whether the specimen belonged to ",
        'Homo sapiens (vs. Pan, Papio, or Pongo), with covariate adjustment for ',
        'estimated age at death ("pop"), sex (probability male from "stdev_age", ',
        "dichotomized at 0.5), and tooth position within the mouth (Anterior, ",
        "Posterior, Premolar from the \"sockets\" column). ",
        f"The estimated coefficient for the Homo sapiens indicator was {coef_human:.3f}, ",
        f"corresponding to an odds ratio of {or_human:.2f} for AMTL compared to non-human primates, ",
        f"with p-value {pval_human:.3g}, which is {sig_text}. ",
    ]

    if yes_no == "Yes":
        explanation_parts.append(
            "Because the Homo sapiens effect is positive and statistically significant, "
            "there is evidence that modern humans have higher AMTL frequencies than "
            "the non-human primate genera after accounting for age, sex, and tooth class. "
        )
    else:
        explanation_parts.append(
            "Because the Homo sapiens effect is not a statistically significant positive "
            "predictor of AMTL, there is not strong evidence that modern humans have "
            "higher AMTL frequencies than the non-human primates once age, sex, and "
            "tooth class are controlled for. "
        )

    explanation_parts.append(
        f"On a 0–100 Likert scale where higher values represent stronger evidence "
        f"for a \"Yes\" answer, I assign a score of {score:d}, which corresponds "
        f"to a \"{yes_no}\" answer to the research question."
    )

    explanation = "".join(explanation_parts)

    conclusion = {
        "response": int(score),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

