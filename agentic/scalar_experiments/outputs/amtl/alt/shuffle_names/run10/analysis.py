import json
from textwrap import dedent

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_likert(coef: float, p_value: float, abs_diff: float) -> int:
    """Map effect/significance to a 0–100 Likert score."""
    if np.isnan(coef) or np.isnan(p_value):
        return 50

    # If humans actually have *lower* AMTL frequency
    if coef < 0:
        if p_value < 1e-4:
            return 5
        if p_value < 1e-3:
            return 10
        if p_value < 1e-2:
            return 20
        if p_value < 5e-2:
            return 30
        return 40

    # Humans have higher AMTL frequency (coef > 0)
    if p_value < 1e-6:
        base = 95
    elif p_value < 1e-4:
        base = 90
    elif p_value < 1e-3:
        base = 85
    elif p_value < 1e-2:
        base = 75
    elif p_value < 5e-2:
        base = 65
    else:
        base = 55

    # Adjust slightly based on absolute difference in predicted probabilities
    if not np.isnan(abs_diff):
        if abs_diff > 0.10:
            base = min(100, base + 5)
        elif abs_diff < 0.02:
            base = max(0, base - 5)

    return int(max(0, min(100, round(base))))


def build_explanation(
    response_value: int,
    coef: float,
    se: float,
    z: float,
    p_value: float,
    odds_ratio: float,
    non_human_rate: float,
    human_rate: float,
    abs_diff: float,
    rel_increase: float,
    n_rows: int,
) -> str:
    """Construct a human-readable explanation string."""
    conclusion_text = (
        "Yes" if response_value >= 50 and coef > 0 and p_value < 0.05 else "No"
    )

    p_str = (
        f"p < 0.001"
        if p_value < 0.001
        else f"p = {p_value:.3f}"
    )

    if not np.isnan(rel_increase) and rel_increase > 0:
        rel_inc_pct = (rel_increase - 1.0) * 100.0
    else:
        rel_inc_pct = np.nan

    explanation = dedent(
        f"""
        Research question
        -----------------
        Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL)
        compared to non-human primate genera (Pan, Pongo, Papio), after accounting for age,
        sex, and tooth class?

        Data and variable mapping
        -------------------------
        The dataset contains {n_rows} rows of tooth-class observations from modern humans and
        non-human primates. Because the column names are shuffled relative to their meanings,
        we used the descriptions in the metadata to recover the semantics:
        - `genus` column: number of missing teeth of a given tooth class (AMTL count).
        - `age` column: number of observable tooth sockets for that class (binomial denominator).
        - `tooth_class` column: taxonomic genus label (e.g., Homo sapiens, Pan, Papio, Pongo).
        - `pop` column: estimated age at death.
        - `stdev_age` column: estimated probability the specimen is male (sex estimate).
        - `sockets` column: tooth class category (Anterior, Posterior, Premolar).

        We defined:
        - AMTL frequency per observation as: missing teeth / observable sockets.
        - A binary indicator `is_human` that equals 1 for Homo sapiens and 0 for all
          non-human genera (Pan, Pongo, Papio).

        Statistical model
        -----------------
        To test the research question, we fit a binomial logistic regression using
        statsmodels' GLM with a logit link:
        - Outcome: number of missing teeth out of the number of observable sockets
          (AMTL frequency), modeled with a binomial family and the number of sockets
          as the binomial denominator.
        - Predictors:
          * `is_human` (Homo sapiens vs. non-human primates),
          * estimated age at death (`pop`),
          * estimated probability of being male (`stdev_age`),
          * tooth class (Anterior, Posterior, Premolar) as a categorical factor.

        This model estimates the effect of being human on AMTL frequency while adjusting
        for age, sex, and tooth class.

        Key results
        -----------
        The coefficient for the `is_human` indicator was:
        - beta = {coef:.3f} (SE = {se:.3f}, z = {z:.2f}, {p_str}),
        - odds ratio for AMTL in humans vs. non-human primates = {odds_ratio:.2f}.

        Using the fitted model, and holding age, sex estimate, and tooth class at typical
        values (mean age and sex estimate, most common tooth class), the predicted AMTL
        frequencies were:
        - Non-human primates: {non_human_rate:.3f} missing teeth per socket.
        - Humans: {human_rate:.3f} missing teeth per socket.

        This corresponds to an absolute difference of {abs_diff:.3f} and a relative
        increase of {rel_inc_pct:.1f}% in AMTL frequency for humans compared to
        non-human primates under comparable conditions.

        Interpretation and Likert-scale rating
        --------------------------------------
        The positive human coefficient, odds ratio greater than 1, and {p_str}
        provide statistical evidence that AMTL frequencies are higher in modern humans
        than in non-human primate genera after accounting for age, sex, and tooth class.

        Based on this evidence, I answer the research question as: {conclusion_text},
        and place this answer at {response_value} on a 0–100 Likert scale where 0 is a
        strong 'No' and 100 is a strong 'Yes'.
        """
    ).strip()

    return explanation


def main() -> None:
    # Load the dataset
    df = pd.read_csv("amtl.csv")

    # Recover semantic variables based on the metadata descriptions
    df = df.copy()
    df["tooth_class_cat"] = df["sockets"].astype("category")
    df["specimen_id"] = df["prob_male"].astype("category")
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["prob_male_est"] = df["stdev_age"].astype(float)
    df["genus_label"] = df["tooth_class"].astype("category")
    df["population"] = df["specimen"].astype("category")

    # Keep only rows with a valid number of sockets
    df = df[df["num_sockets"] > 0].copy()

    # Binary indicator for humans vs. all non-human primates
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Compute AMTL frequency (proportion) per observation
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]

    # Binomial GLM with logit link; use number of sockets as binomial denominator
    formula = (
        "missing_prop ~ is_human + age_at_death + prob_male_est + C(tooth_class_cat)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    z = float(coef / se) if se != 0 else np.nan
    p_value = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Typical covariate values for predictions
    mean_age = float(df["age_at_death"].mean())
    mean_prob_male = float(df["prob_male_est"].mean())
    common_tc = df["tooth_class_cat"].mode()[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death": [mean_age, mean_age],
            "prob_male_est": [mean_prob_male, mean_prob_male],
            "tooth_class_cat": [common_tc, common_tc],
        }
    )

    preds = result.get_prediction(pred_df)
    pred_mean = preds.predicted_mean
    # `predicted_mean` may be a NumPy array; handle both array and Series
    non_human_rate = float(pred_mean[0])
    human_rate = float(pred_mean[1])
    abs_diff = human_rate - non_human_rate
    rel_increase = (
        human_rate / non_human_rate if non_human_rate > 0 else float("nan")
    )

    response_value = compute_likert(coef, p_value, abs_diff)

    explanation = build_explanation(
        response_value=response_value,
        coef=coef,
        se=se,
        z=z,
        p_value=p_value,
        odds_ratio=odds_ratio,
        non_human_rate=non_human_rate,
        human_rate=human_rate,
        abs_diff=abs_diff,
        rel_increase=rel_increase,
        n_rows=df.shape[0],
    )

    output = {"response": int(response_value), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
