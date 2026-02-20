import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Create human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categorical variables are treated as such
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Use a Poisson regression with log link and log(sockets) as an exposure offset
    # to model AMTL rates per observable socket.
    sockets = df["sockets"].to_numpy()
    offset = np.log(sockets)

    model = smf.glm(
        formula="num_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    )
    result = model.fit()

    # Predicted AMTL rates for humans vs non-humans,
    # evaluated over the empirical distribution of age, sex, tooth class, and exposure.
    human_data = df.copy()
    human_data["is_human"] = 1
    human_pred_counts = result.predict(human_data, offset=offset)
    human_rate = float(human_pred_counts.sum() / sockets.sum())

    nonhuman_data = df.copy()
    nonhuman_data["is_human"] = 0
    nonhuman_pred_counts = result.predict(nonhuman_data, offset=offset)
    nonhuman_rate = float(nonhuman_pred_counts.sum() / sockets.sum())

    diff = human_rate - nonhuman_rate

    # Extract coefficient information for is_human
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    pval = float(result.pvalues["is_human"])

    # 95% confidence interval on log-rate-ratio scale
    z_95 = 1.96
    ci_low = coef - z_95 * se
    ci_high = coef + z_95 * se

    # Map evidence to a 0-100 Likert-style score
    # Strong positive, precise, and meaningful effect -> closer to 100
    # Weak/uncertain effect -> around 50
    # Negative effect (humans lower AMTL) -> closer to 0
    if diff > 0 and pval < 0.001:
        base_score = 90
    elif diff > 0 and pval < 0.01:
        base_score = 80
    elif diff > 0 and pval < 0.05:
        base_score = 70
    elif diff > 0 and pval < 0.1:
        base_score = 60
    elif diff > 0:
        base_score = 55
    elif abs(diff) <= 0.005 and pval >= 0.1:
        base_score = 50
    elif diff < 0 and pval < 0.05:
        base_score = 30
    elif diff < 0:
        base_score = 40
    else:
        base_score = 50

    # Further adjust score based on magnitude of difference in absolute AMTL rate
    # (so that trivially small but significant effects do not get extreme scores).
    abs_diff = abs(diff)
    if abs_diff > 0.08:
        magnitude_adj = 8
    elif abs_diff > 0.05:
        magnitude_adj = 5
    elif abs_diff > 0.02:
        magnitude_adj = 3
    else:
        magnitude_adj = 0

    if diff > 0:
        score = base_score + magnitude_adj
    elif diff < 0:
        score = base_score - magnitude_adj
    else:
        score = base_score

    score_int = int(min(max(round(score), 0), 100))

    # Build explanation text
    rate_ratio = float(np.exp(coef))
    rate_ci_low = float(np.exp(ci_low))
    rate_ci_high = float(np.exp(ci_high))

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?\n\n"
        "Data and model: I analyzed 1,450 observations from amtl.csv. For each specimen "
        "and tooth class, I modeled the count of missing teeth (num_amtl) using a "
        "Poisson regression with a log link and log(sockets) as an exposure offset. "
        "This specification estimates AMTL rates per observable socket. The main "
        "predictor was a binary indicator for Homo sapiens versus all non-human primates "
        "combined, with age at death, estimated probability of being male (prob_male), "
        "and tooth class (anterior, premolar, posterior) included as covariates.\n\n"
        f"Results: The estimated mean AMTL rate (expected number of missing teeth per "
        f"observable socket) for modern humans, averaged over the observed distribution "
        f"of age, sex, tooth class, and exposure, is approximately {human_rate:.3f}. For "
        f"non-human primates (Pan, Pongo, Papio), the corresponding predicted rate is "
        f"approximately {nonhuman_rate:.3f}. This implies an absolute difference of "
        f"about {diff:.3f} (human minus non-human).\n\n"
        f"In the Poisson regression, the coefficient for the human indicator on the "
        f"log-rate scale is {coef:.3f} with standard error {se:.3f} and p-value "
        f"{pval:.3g}. The corresponding rate ratio for AMTL in humans versus non-human "
        f"primates is {rate_ratio:.3f}, with an approximate 95% confidence interval from "
        f"{rate_ci_low:.3f} to {rate_ci_high:.3f}. "
        "Thus, the model suggests that humans have "
        + ("higher" if diff > 0 else "lower" if diff < 0 else "similar")
        + " AMTL rates compared to non-human primates after controlling for age, sex, "
        "and tooth class.\n\n"
        "Interpretation: Based on this model, the direction of the estimated effect "
        "and its statistical significance indicate that "
        + (
            "modern humans do exhibit higher AMTL frequencies than non-human primates"
            if diff > 0
            else "modern humans do not exhibit higher AMTL frequencies than non-human primates"
            if diff < 0
            else "there is no clear difference in AMTL frequencies between modern humans and non-human primates"
        )
        + " under the specified controls. The assigned Likert-scale score reflects the "
        "strength and precision of this evidence, taking into account both the p-value "
        "for the human effect and the magnitude of the difference in predicted AMTL "
        "rates."
    )

    conclusion = {
        "response": score_int,
        "explanation": explanation,
    }

    # Write strict JSON to conclusion.txt
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
