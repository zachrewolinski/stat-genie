import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL at the individual-socket level with:
    - response: amtl (1 if the socket shows antemortem tooth loss, 0 otherwise)
    - predictors: is_human, age, prob_male, tooth_class
    - cluster-robust SEs by specimen
    The input DataFrame is expected to have one row per socket.
    """
    model = smf.glm(
        formula="amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})
    return result


def compute_likert_score(coef: float, pval: float, odds_ratio: float) -> int:
    """
    Map the evidence about the human effect to a 0-100 Likert score.

    0   -> strong "No" (humans clearly lower)
    50  -> no clear evidence either way
    100 -> strong "Yes" (humans clearly higher)
    """
    # Start from neutral evidence.
    score = 50.0

    # Direction and strength from odds ratio.
    if coef > 0:
        # Humans have higher AMTL in the model.
        if pval < 0.001:
            score = 90.0
        elif pval < 0.01:
            score = 80.0
        elif pval < 0.05:
            score = 70.0
        elif pval < 0.1:
            score = 60.0
        else:
            score = 55.0

        # Adjust slightly based on effect size.
        if odds_ratio > 2.0:
            score += 5.0
        if odds_ratio > 3.0:
            score += 5.0
    else:
        # coef <= 0: humans are similar or lower.
        if pval < 0.001:
            score = 10.0
        elif pval < 0.01:
            score = 15.0
        elif pval < 0.05:
            score = 20.0
        elif pval < 0.1:
            score = 35.0
        else:
            score = 45.0

        # Stronger protective effect (lower odds) pushes further toward 0.
        if odds_ratio < 0.5:
            score -= 5.0
        if odds_ratio < 0.33:
            score -= 5.0

    # Clamp to [0, 100] and convert to int.
    return int(max(0, min(100, round(score))))


def build_explanation(
    result,
    coef: float,
    pval: float,
    ci_low: float,
    ci_high: float,
    odds_ratio: float,
    or_ci_low: float,
    or_ci_high: float,
    human_prob: float,
    nonhuman_prob: float,
    n_total: int,
    n_used: int,
    n_invalid_ratio: int,
) -> str:
    """
    Create a human-readable explanation summarizing model, evidence, and conclusion.
    """
    lines = []
    lines.append(
        "I modeled antemortem tooth loss (AMTL) as a binomial outcome at the "
        "individual tooth-socket level by expanding each specimen × tooth-class "
        "record into one row per observable socket, with a binary indicator "
        "(`amtl`) marking whether that socket was lost ante mortem."
    )
    if n_invalid_ratio > 0:
        lines.append(
            f"Out of {n_total} total specimen × tooth-class observations, "
            f"I excluded {n_invalid_ratio} rows where the recorded number of "
            "missing teeth exceeded the number of observable sockets, since "
            "these represent inconsistent counts for a binomial model. "
            f"The final analysis therefore used {n_used} valid observations."
        )
    else:
        lines.append(
            f"The analysis used all {n_used} specimen × tooth-class "
            "observations after basic cleaning (removing rows with missing "
            "values and ensuring a positive number of sockets)."
        )
    lines.append(
        "I then fit a binomial logistic regression with predictors: "
        "an indicator for modern humans vs. non-human primates (`is_human`), "
        "estimated age at death (`age`), probability of being male "
        "(`prob_male`), and tooth class (`tooth_class` as a categorical factor). "
        "To account for non-independence of sockets within the same individual, "
        "I used cluster-robust standard errors at the specimen level."
    )

    lines.append(
        f"The coefficient for modern humans (`is_human`) on the log-odds scale "
        f"was {coef:.3f} with p-value {pval:.3g}, giving an odds ratio of "
        f"{odds_ratio:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}). "
        f"The 95% confidence interval for the log-odds coefficient was "
        f"{ci_low:.3f} to {ci_high:.3f}."
    )

    lines.append(
        "To make this more interpretable, I computed predicted probabilities of "
        "AMTL for a typical case (mean age, mean sex probability, and the most "
        "common tooth class) for humans versus non-human primates. "
        f"In this scenario, the estimated AMTL probability was "
        f"{nonhuman_prob:.3f} for non-human primates and "
        f"{human_prob:.3f} for modern humans."
    )

    if coef > 0:
        direction_text = (
            "This indicates that, after accounting for age, sex, and tooth "
            "class, modern humans tend to have higher AMTL frequencies than "
            "non-human primates."
        )
    elif coef < 0:
        direction_text = (
            "This indicates that, after accounting for age, sex, and tooth "
            "class, modern humans tend to have similar or lower AMTL "
            "frequencies than non-human primates."
        )
    else:
        direction_text = (
            "This indicates no detectable difference in AMTL frequencies "
            "between modern humans and non-human primates once age, sex, and "
            "tooth class are accounted for."
        )

    lines.append(direction_text)

    if pval < 0.05:
        significance_text = (
            "The effect of the human indicator is statistically significant at "
            "the 5% level, so the data provide evidence for a real difference "
            "in AMTL frequencies between humans and non-human primates under "
            "this model."
        )
    elif pval < 0.1:
        significance_text = (
            "The effect of the human indicator is only marginally significant "
            "at conventional thresholds, so the evidence for a difference in "
            "AMTL frequencies is suggestive but not strong."
        )
    else:
        significance_text = (
            "The effect of the human indicator is not statistically significant "
            "at conventional levels, so the data do not provide clear evidence "
            "for a difference in AMTL frequencies between humans and "
            "non-human primates once age, sex, and tooth class are controlled."
        )

    lines.append(significance_text)

    if coef > 0:
        conclusion_sentence = (
            "Overall, the binomial regression suggests that modern humans "
            "have higher AMTL frequencies than non-human primates after "
            "accounting for age, sex, and tooth class; the Likert score "
            "reflects this as a 'Yes' leaning conclusion, with strength "
            "proportional to the statistical significance and effect size."
        )
    elif coef < 0:
        conclusion_sentence = (
            "Overall, the binomial regression does not support the claim that "
            "modern humans have higher AMTL frequencies than non-human primates "
            "after accounting for age, sex, and tooth class; if anything, the "
            "point estimates suggest similar or lower frequencies in humans, "
            "and the Likert score reflects a 'No' leaning conclusion whose "
            "strength depends on the statistical evidence."
        )
    else:
        conclusion_sentence = (
            "Overall, the binomial regression shows no meaningful difference "
            "in AMTL frequencies between modern humans and non-human primates, "
            "once age, sex, and tooth class are accounted for; the Likert "
            "score therefore remains near the neutral 'No evidence for a "
            "difference' region."
        )

    lines.append(conclusion_sentence)

    return " ".join(lines)


def main():
    data_path = Path("amtl.csv")
    df_raw = pd.read_csv(data_path)

    n_total = int(df_raw.shape[0])

    # Exclude rows with logically inconsistent AMTL counts.
    invalid_ratio_mask = df_raw["num_amtl"] > df_raw["sockets"]
    n_invalid_ratio = int(invalid_ratio_mask.sum())
    df = df_raw.loc[~invalid_ratio_mask].copy()

    # Basic cleaning and derived variables.
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "tooth_class",
            "genus",
            "specimen",
        ]
    )

    # Ensure positive number of sockets.
    df = df[df["sockets"] > 0]

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = df["genus"].astype(str).str.startswith("Homo").astype(int)
    n_used = int(df.shape[0])

    # Expand to one row per socket with a binary AMTL indicator.
    socket_rows = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_missing = int(row["num_amtl"])
        # Safety: cap at the number of sockets after prior filtering.
        n_missing = max(0, min(n_missing, n_sockets))
        outcomes = np.array([1] * n_missing + [0] * (n_sockets - n_missing), dtype=int)
        base = {
            "age": row["age"],
            "prob_male": row["prob_male"],
            "tooth_class": row["tooth_class"],
            "is_human": row["is_human"],
            "specimen": row["specimen"],
        }
        tmp = pd.DataFrame(base, index=range(n_sockets))
        tmp["amtl"] = outcomes
        socket_rows.append(tmp)

    df_long = pd.concat(socket_rows, ignore_index=True)

    # Fit the binomial model on socket-level data.
    result = fit_binomial_model(df_long)

    # Extract human effect and its uncertainty.
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()

    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted probabilities for a typical case.
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    # Use the most common tooth class to define a typical dental context.
    common_tooth_class = df["tooth_class"].mode().iloc[0]

    exog = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )

    pred_probs = result.predict(exog)
    nonhuman_prob = float(pred_probs.iloc[0])
    human_prob = float(pred_probs.iloc[1])

    # Likert score summarizing evidence for "Yes, humans have higher AMTL".
    response_score = compute_likert_score(coef, pval, odds_ratio)

    explanation = build_explanation(
        result=result,
        coef=coef,
        pval=pval,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        human_prob=human_prob,
        nonhuman_prob=nonhuman_prob,
        n_total=n_total,
        n_used=n_used,
        n_invalid_ratio=n_invalid_ratio,
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
