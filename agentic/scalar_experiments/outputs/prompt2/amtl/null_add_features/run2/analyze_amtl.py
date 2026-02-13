import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep rows with valid socket counts
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Outcome: binomial counts (successes = num_amtl, failures = sockets - num_amtl)
    y = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    # Predictors: human vs non-human, age, sex proxy, and tooth class
    X = df[["is_human", "age", "prob_male"]].copy()
    tooth_dummies = pd.get_dummies(
        df["tooth_class"], prefix="tooth", drop_first=True
    )
    X = pd.concat([X, tooth_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    # Effect of being human vs non-human
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    pvalue_human = float(result.pvalues["is_human"])

    # Average marginal difference by toggling is_human while holding other covariates fixed
    X_human = X.copy()
    X_nonhuman = X.copy()
    X_human["is_human"] = 1
    X_nonhuman["is_human"] = 0
    mean_prob_human = float(result.predict(X_human).mean())
    mean_prob_nonhuman = float(result.predict(X_nonhuman).mean())
    diff_prob = mean_prob_human - mean_prob_nonhuman

    # Decide on Yes/No based on direction and strength of effect
    # Positive diff/coef would indicate higher AMTL in humans.
    if diff_prob > 0 and pvalue_human < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic:
    # - If the data clearly support higher AMTL in humans (response == Yes),
    #   tie confidence to the p-value and effect size.
    # - If the data do NOT support higher AMTL in humans (response == No),
    #   and the estimated human effect is small and imprecise (near zero with
    #   a large p-value), we can still be reasonably confident that humans do
    #   not have meaningfully higher AMTL frequencies in this dataset.
    if response == "Yes":
        if pvalue_human < 0.001:
            base_conf = 90
        elif pvalue_human < 0.01:
            base_conf = 80
        elif pvalue_human < 0.05:
            base_conf = 70
        else:
            base_conf = 60
        effect_scale = min(abs(diff_prob) / 0.05, 1.0)
        confidence = int(round(base_conf * (0.5 + 0.5 * effect_scale)))
    else:
        # For a "No" answer, boost confidence when:
        # - the estimated human effect is near zero or negative, and
        # - the p-value is large (no sign of a positive human effect).
        if diff_prob <= 0 and pvalue_human > 0.1 and abs(diff_prob) < 0.02:
            confidence = 85
        else:
            confidence = 70

    confidence = max(0, min(100, int(round(confidence))))

    explanation = (
        "I fit a binomial regression model (GLM with logit link) where the number of "
        "missing teeth (num_amtl) out of observable sockets was modeled as a function "
        "of a human-vs-non-human indicator, age at death, a sex proxy (prob_male), and "
        "tooth class (dummy variables for Premolar and Posterior vs Anterior). "
        f"The coefficient for the human indicator was {coef_human:.3f} with standard "
        f"error {se_human:.3f} (p-value {pvalue_human:.3f}). "
        f"When I computed predicted AMTL frequencies while holding age, sex, and tooth "
        "class constant but toggling the human indicator, the average predicted AMTL "
        f"frequency for humans was {mean_prob_human:.3f}, compared to "
        f"{mean_prob_nonhuman:.3f} for non-human primates, a difference of "
        f"{diff_prob:.3f}. "
        "Because the human effect is not both clearly positive and statistically strong "
        "under this model, I conclude that modern humans do not have higher AMTL "
        "frequencies than non-human primates after accounting for age, sex, and tooth class."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")

    # Also print a short summary to the console for inspection
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
