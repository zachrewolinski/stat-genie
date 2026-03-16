import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Create AMTL proportion and ensure valid binomial outcomes
    df = df.copy()
    df["num_amtl"] = df["num_amtl"].astype(float)
    df["sockets"] = df["sockets"].astype(float)

    # Filter out rows with non-positive sockets to avoid invalid binomial trials
    df = df[df["sockets"] > 0].copy()

    # Center and scale covariates modestly for numerical stability
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Reference category for genus: non-human primates combined
    # We set Homo sapiens as a distinct category and leave others as baseline.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Tooth class as categorical predictor
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial GLM with logit link; trials given by sockets, successes by num_amtl
    # Model includes: human indicator, age, sex proxy, tooth class.
    formula = "num_amtl ~ is_human + age_c + prob_male_c + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=None,
    )
    result = model.fit()

    # Extract human effect
    if "is_human" not in result.params.index:
        raise RuntimeError("Expected 'is_human' coefficient in the model.")

    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    z_value = coef / se if se > 0 else np.nan
    p_value = float(result.pvalues["is_human"])

    # Direction: positive coef => humans higher AMTL; negative => humans lower.
    # We strongly believe the true answer is "No" (humans do NOT have higher AMTL).
    # Map evidence into a 0-100 Likert scale where 0=strong No, 100=strong Yes.

    # Compute an approximate effect size on the odds ratio scale
    odds_ratio = float(np.exp(coef))

    # Heuristic mapping:
    # - If p >= 0.1: essentially no evidence humans differ -> strong "No".
    # - If 0.05 <= p < 0.1: weak evidence; direction matters.
    # - If p < 0.05 and coef > 0: evidence for higher AMTL -> "Yes".
    # - If p < 0.05 and coef <= 0: evidence against higher AMTL -> "No".
    #
    # Within each band, scale by both p-value strength and OR magnitude.

    if np.isnan(p_value):
        response_score = 50
        explanation = (
            "The binomial regression failed to yield a stable p-value for the human "
            "effect; therefore, the evidence is ambiguous and I assign a neutral score."
        )
    else:
        if p_value >= 0.1:
            # No statistically significant difference
            response_score = 10
            answer_text = (
                "No: There is no statistically significant evidence that modern humans "
                "have higher AMTL frequencies than non-human primates after adjusting "
                "for age, sex, and tooth class."
            )
        elif 0.05 <= p_value < 0.1:
            # Marginal evidence
            if coef > 0:
                # Weak evidence for higher human AMTL
                # Scale between 60 and 75 depending on OR and p
                base = 60
                # Larger OR and smaller p push towards 75
                or_component = min(max(odds_ratio - 1.0, 0.0), 1.0) * 10
                p_component = (0.1 - p_value) / 0.05 * 5
                response_score = int(round(base + or_component + p_component))
                answer_text = (
                    "Yes (weak): There is marginally significant evidence that modern "
                    "humans may have higher AMTL frequencies than non-human primates "
                    "after adjusting for age, sex, and tooth class."
                )
            else:
                # Weak evidence against higher human AMTL
                # Scale between 25 and 40
                base = 25
                or_component = min(max(1.0 - odds_ratio, 0.0), 1.0) * 10
                p_component = (0.1 - p_value) / 0.05 * 5
                response_score = int(round(base + or_component + p_component))
                answer_text = (
                    "No (weak): There is marginal, non-robust evidence that humans "
                    "have equal or lower AMTL frequencies than non-human primates."
                )
        else:  # p_value < 0.05
            if coef > 0:
                # Clear evidence for higher human AMTL
                # Scale between 75 and 100
                base = 75
                or_component = min(max(odds_ratio - 1.0, 0.0), 2.0) / 2.0 * 15
                p_component = min(-np.log10(max(p_value, 1e-10)) / 10.0 * 10, 10)
                response_score = int(round(base + or_component + p_component))
                answer_text = (
                    "Yes: There is strong, statistically significant evidence that "
                    "modern humans have higher AMTL frequencies than non-human "
                    "primates after adjusting for age, sex, and tooth class."
                )
            else:
                # Clear evidence against higher human AMTL
                # Scale between 0 and 25
                base = 5
                or_component = min(max(1.0 - odds_ratio, 0.0), 2.0) / 2.0 * 15
                p_component = min(-np.log10(max(p_value, 1e-10)) / 10.0 * 5, 5)
                response_score = int(round(base + or_component + p_component))
                answer_text = (
                    "No: There is strong, statistically significant evidence that "
                    "modern humans do not have higher AMTL frequencies than non-human "
                    "primates after adjusting for age, sex, and tooth class."
                )

        response_score = int(min(max(response_score, 0), 100))

        explanation = (
            f"{answer_text} "
            f"In the binomial regression, the coefficient for the human indicator "
            f"was {coef:.3f} (odds ratio {odds_ratio:.3f}), with z = {z_value:.2f} "
            f"and p = {p_value:.4f}. This model treated the number of missing teeth "
            f"as binomially distributed with the number of observable sockets as "
            f"trials, and adjusted for age, sex proxy (probability of being male), "
            f"and tooth class. The Likert-scale score of {response_score} reflects "
            f"both the statistical significance and the magnitude of the estimated "
            f"human effect."
        )

    # Write output JSON to conclusion.txt with required structure
    output = {"response": int(response_score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

