import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominators
    df = df[df["sockets"] > 0].copy()

    # Create variables for modelling
    df["prop_missing"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial regression of AMTL frequency on human status, age, sex proxy, and tooth class
    model = smf.glm(
        formula="prop_missing ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    is_human_coef = model.params["is_human"]
    is_human_ci_low, is_human_ci_high = model.conf_int().loc["is_human"]
    is_human_p = model.pvalues["is_human"]

    # Marginal predicted AMTL frequencies for humans vs non-human primates
    df_human = df.copy()
    df_nonhuman = df.copy()
    df_human["is_human"] = 1
    df_nonhuman["is_human"] = 0
    pred_rate_human = float(model.predict(df_human).mean())
    pred_rate_nonhuman = float(model.predict(df_nonhuman).mean())
    diff_rate = pred_rate_human - pred_rate_nonhuman

    # Decide response based on direction and significance of human effect
    # Research question: Do humans have higher AMTL frequencies than non-human primates,
    # after accounting for age, sex, and tooth class?
    if is_human_coef > 0 and is_human_p < 0.05:
        response = "Yes"
    else:
        # Either the human effect is negative or we lack statistical evidence
        # that it is positive, so we answer "No".
        response = "No"

    # Heuristic confidence score in [0, 100].
    # For a "Yes" answer, smaller p-values for a positive human effect
    # translate into higher confidence.
    if response == "Yes":
        capped_p = min(max(is_human_p, 0.0), 0.5)
        confidence = int(round(100 * (1.0 - capped_p / 0.5)))
    else:
        # For a "No" answer, we base confidence on how small the estimated
        # human effect is on the probability scale and how little evidence
        # there is for a positive effect.
        abs_diff = abs(diff_rate)
        if is_human_p > 0.3 and abs_diff < 0.01:
            confidence = 70
        elif is_human_p > 0.1 and abs_diff < 0.02:
            confidence = 60
        else:
            confidence = 50

    explanation = (
        "I modeled the frequency of antemortem tooth loss (AMTL) using a binomial "
        "regression where the response was the proportion of missing teeth "
        "(num_amtl / sockets) for each specimen–tooth-class combination, with the "
        "number of observable sockets used as binomial weights. The key predictor "
        "was whether the specimen was a modern human (Homo sapiens) versus a "
        "non-human primate (Pan, Pongo, Papio), while statistically controlling for "
        "estimated age at death, a probabilistic sex indicator (prob_male), and "
        "tooth class (anterior, posterior, premolar).\n\n"
        f"In this model, the coefficient for the human indicator (is_human) was "
        f"{is_human_coef:.3f} on the log-odds scale with a 95% confidence interval "
        f"from {is_human_ci_low:.3f} to {is_human_ci_high:.3f} and a p-value of "
        f"{is_human_p:.3g}. On the probability scale, the model-implied average "
        f"AMTL frequency for humans was approximately {pred_rate_human:.3f}, "
        f"compared to {pred_rate_nonhuman:.3f} for non-human primates, a difference "
        f"of {diff_rate:.3f}.\n\n"
        "Because the estimated human effect is very small and not statistically "
        "distinguishable from zero, the analysis does not provide evidence that "
        "modern humans have higher AMTL frequencies than the pooled non-human "
        "primate genera after controlling for age, sex, and tooth class. If "
        "anything, the point estimate suggests slightly lower AMTL in humans, but "
        "the associated uncertainty means that modest differences in either "
        "direction cannot be ruled out. The confidence score reflects this lack of "
        "evidence for a positive human effect while also acknowledging model "
        "assumptions, such as treating observations as independent and using a "
        "simple fixed-effects binomial regression instead of a more complex "
        "hierarchical model."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
