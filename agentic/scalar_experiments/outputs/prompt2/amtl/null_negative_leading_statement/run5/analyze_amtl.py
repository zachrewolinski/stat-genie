import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominators
    df = df[df["sockets"] > 0].copy()

    # Create proportion of missing teeth and human indicator
    df["amtl_prop"] = df["num_amtl"] / df["sockets"].astype(float)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Fit binomial regression with weights equal to number of sockets
    model = smf.glm(
        "amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract effect of being human
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    odds_ratio = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Decide answer: only say "Yes" if human effect is significantly positive
    if coef > 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic based on p-value and whether CI excludes 1
    if pval < 1e-6:
        base_conf = 95
    elif pval < 1e-3:
        base_conf = 90
    elif pval < 0.05:
        base_conf = 80
    elif pval < 0.2:
        base_conf = 65
    else:
        base_conf = 55

    if or_low < 1.0 < or_high:
        base_conf = min(base_conf, 60)

    confidence = int(base_conf)

    # Build explanation string describing model and key results
    explanation_parts = [
        (
            "I analyzed the AMTL dataset using a binomial logistic regression, "
            "modeling the probability of antemortem tooth loss (num_amtl / sockets) "
            "with a binary indicator for modern humans (Homo sapiens) versus non-human primates, "
            "while controlling for age at death, estimated sex (prob_male), and tooth class "
            "(Anterior, Posterior, Premolar)."
        ),
        (
            f" The estimated log-odds coefficient for the human indicator is {coef:.3f}, "
            f"corresponding to an odds ratio of {odds_ratio:.2f} with a 95% confidence interval "
            f"from {or_low:.2f} to {or_high:.2f} (p = {pval:.3g})."
        ),
    ]

    if response == "Yes":
        explanation_parts.append(
            " Because the human indicator is positive and statistically significant (p < 0.05), "
            "this suggests that, after adjusting for age, sex, and tooth class, modern humans "
            "have higher frequencies of antemortem tooth loss than the pooled non-human primate genera."
        )
    else:
        if coef < 0 and pval < 0.05:
            explanation_parts.append(
                " The human indicator is significantly negative, indicating that modern humans actually "
                "have lower AMTL frequencies than non-human primates once covariates are controlled, "
                "so we reject the hypothesis that humans have higher AMTL."
            )
        else:
            explanation_parts.append(
                " The human indicator is not significantly positive at the 0.05 level, so the data do not "
                "provide strong evidence that modern humans have higher AMTL frequencies than non-human primates "
                "after accounting for age, sex, and tooth class."
            )

    explanation = "".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

