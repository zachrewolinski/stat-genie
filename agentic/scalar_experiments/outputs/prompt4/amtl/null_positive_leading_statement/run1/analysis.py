import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Derived variables
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop any rows with sockets <= 0 just in case
    df = df[df["sockets"] > 0].copy()

    # Design matrix for binomial GLM
    # We treat age and prob_male as continuous, tooth_class as categorical.
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    X = sm.add_constant(X, has_constant="add")

    y = df["amtl_rate"]
    var_weights = df["sockets"]

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        var_weights=var_weights,
    )
    result = model.fit()

    # Extract coefficient and p-value for humans vs non-humans
    coef_human = result.params.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)

    # Determine direction and strength of evidence
    if np.isnan(coef_human) or np.isnan(pvalue_human):
        response_score = 50
        explanation = (
            "The binomial regression model could not reliably estimate the effect of "
            "modern humans versus non-human primates on antemortem tooth loss. "
            "Therefore, the data are inconclusive with respect to the research question."
        )
    else:
        # Compute an approximate odds ratio for humans vs non-humans
        odds_ratio = float(np.exp(coef_human))

        # Map statistical evidence to a 0–100 scale
        if coef_human > 0 and pvalue_human < 0.001:
            response_score = 95
        elif coef_human > 0 and pvalue_human < 0.01:
            response_score = 85
        elif coef_human > 0 and pvalue_human < 0.05:
            response_score = 75
        elif coef_human > 0:
            response_score = 60
        elif coef_human < 0 and pvalue_human < 0.001:
            response_score = 5
        elif coef_human < 0 and pvalue_human < 0.01:
            response_score = 15
        elif coef_human < 0 and pvalue_human < 0.05:
            response_score = 25
        elif coef_human < 0:
            response_score = 40
        else:
            response_score = 50

        # Build textual explanation summarizing the model and findings
        explanation = (
            "I fit a binomial regression model for the proportion of antemortem tooth "
            "loss (number of missing teeth divided by observable sockets) with genus "
            "grouped as humans versus non-human primates, while adjusting for age, "
            "sex (via the probability of being male), and tooth class. "
            f"The estimated coefficient for modern humans relative to non-human primates "
            f"on the log-odds scale was {coef_human:.3f}, corresponding to an odds ratio "
            f"of approximately {odds_ratio:.2f}. The associated p-value was "
            f"{pvalue_human:.3g}. "
        )

        if coef_human > 0 and pvalue_human < 0.05:
            explanation += (
                "This positive and statistically significant effect indicates that, after "
                "controlling for age, sex, and tooth class, modern humans have higher "
                "frequencies of antemortem tooth loss compared to the pooled group of "
                "non-human primate genera (Pan, Pongo, Papio)."
            )
        elif coef_human < 0 and pvalue_human < 0.05:
            explanation += (
                "This negative and statistically significant effect indicates that, after "
                "controlling for age, sex, and tooth class, modern humans have lower "
                "frequencies of antemortem tooth loss compared to the pooled group of "
                "non-human primate genera (Pan, Pongo, Papio)."
            )
        else:
            explanation += (
                "Because the estimated human effect is not statistically distinguishable "
                "from zero at conventional significance levels, the data do not provide "
                "strong evidence that modern humans differ from non-human primates in "
                "frequencies of antemortem tooth loss after adjusting for age, sex, and "
                "tooth class."
            )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

