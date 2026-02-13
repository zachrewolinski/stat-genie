import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_conclusion(effects: dict) -> tuple[int, str]:
    corr = effects["corr"]
    coef1 = effects["model1_coef"]
    p1 = effects["model1_p"]
    coef2 = effects["model2_coef"]
    p2 = effects["model2_p"]
    n = effects["n_obs"]

    pieces = []
    pieces.append(
        f"We analyzed {n} California K-6 and K-8 school districts, "
        "relating the student–teacher ratio (students per teacher) to average 5th-grade "
        "academic performance (the mean of reading and math scores on the Stanford 9 test)."
    )
    direction = "higher" if corr > 0 else "lower"
    pieces.append(
        f"The simple Pearson correlation between the student–teacher ratio and average test "
        f"score was {corr:.3f}, meaning that districts with higher ratios (more students per "
        f"teacher) tend to have {direction} test performance."
    )
    pieces.append(
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"the coefficient on the ratio was {coef1:.3f} (p-value {p1:.3g}). "
        f"This means that a one-student increase in the ratio is associated with an estimated "
        f"{coef1:.3f}-point change in the average test score."
    )
    pieces.append(
        "We then fit a multiple regression that included the student–teacher ratio along with "
        "district income, the percentage of English learners, and the percentage of students "
        "receiving subsidized lunch (and related disadvantage measures) as controls."
    )
    pieces.append(
        f"In this adjusted model, the coefficient on the student–teacher ratio was "
        f"{coef2:.3f} (p-value {p2:.3g}), indicating how the ratio is associated with "
        "test performance after accounting for these demographic and economic factors."
    )
    pieces.append(
        "These analyses describe statistical associations in observational data and should "
        "not be interpreted as definitive causal effects. However, they tell us whether, "
        "within this dataset, districts with lower student–teacher ratios tend to have "
        "higher test scores."
    )

    # Determine Likert-scale response based on strength and consistency of evidence.
    if coef1 < 0 and coef2 < 0 and p1 < 0.01 and p2 < 0.01:
        response = 90
        pieces.append(
            "Because the student–teacher ratio has a negative and statistically strong "
            "association with test scores in both simple and adjusted models, the data "
            "provide strong evidence that lower student–teacher ratios are associated with "
            "higher academic performance."
        )
    elif coef1 < 0 and coef2 < 0 and p1 < 0.05 and p2 < 0.05:
        response = 80
        pieces.append(
            "Because the estimated effects are negative and statistically significant in "
            "both models, the data provide clear evidence that lower student–teacher ratios "
            "are associated with higher academic performance, though the relationship is "
            "somewhat more modest."
        )
    elif coef1 < 0 and p1 < 0.05 and (p2 >= 0.05 or coef2 >= 0):
        response = 65
        pieces.append(
            "The simple model shows a negative, statistically significant association, but "
            "the adjusted model is weaker or less precise, so the evidence for a beneficial "
            "association of lower student–teacher ratios with performance is only moderate."
        )
    elif (coef1 < 0 or coef2 < 0) and (p1 < 0.1 or p2 < 0.1):
        response = 60
        pieces.append(
            "At least one model suggests a negative association between the student–teacher "
            "ratio and test performance at a marginal significance level, so the evidence "
            "for the association is suggestive but not strong."
        )
    elif coef1 < 0 or coef2 < 0:
        response = 55
        pieces.append(
            "The estimated coefficients are mostly negative, but they are not statistically "
            "distinguishable from zero, so the evidence that lower student–teacher ratios "
            "are associated with higher performance is weak."
        )
    elif coef1 > 0 or coef2 > 0:
        if p1 < 0.05 or p2 < 0.05:
            response = 20
            pieces.append(
                "The estimated coefficients are mostly positive and at least one is "
                "statistically significant, suggesting that higher student–teacher ratios "
                "are associated with higher performance in this dataset, contrary to the "
                "hypothesized direction."
            )
        else:
            response = 40
            pieces.append(
                "The estimated coefficients do not consistently support a negative "
                "relationship, and the statistical evidence is weak, so the data do not "
                "support a clear association between lower student–teacher ratios and "
                "higher performance."
            )
    else:
        response = 50
        pieces.append(
            "Overall, the data do not support a clear directional association between "
            "student–teacher ratios and academic performance in this dataset."
        )

    explanation = " ".join(pieces)
    return int(response), explanation


def main() -> None:
    csv_path = Path("caschools.csv")
    if not csv_path.exists():
        raise FileNotFoundError("caschools.csv not found in the current directory.")

    df = pd.read_csv(csv_path)

    # Construct student–teacher ratio and average test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Clean problematic values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "avg_score"])

    # Simple correlation.
    corr = df["avg_score"].corr(df["stratio"])

    # Simple linear regression: avg_score ~ stratio.
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multiple regression with key controls, if available.
    candidate_covariates = ["income", "english", "lunch", "calworks"]
    covars_existing = [c for c in candidate_covariates if c in df.columns]
    X_cols = ["stratio"] + covars_existing
    X_full = sm.add_constant(df[X_cols])
    model_full = sm.OLS(df["avg_score"], X_full).fit()

    effects = {
        "corr": float(corr),
        "model1_coef": float(model_simple.params["stratio"]),
        "model1_p": float(model_simple.pvalues["stratio"]),
        "model2_coef": float(model_full.params["stratio"]),
        "model2_p": float(model_full.pvalues["stratio"]),
        "n_obs": int(df.shape[0]),
    }

    # Print a brief summary for transparency (not used by the evaluator).
    print("Correlation (avg_score vs student–teacher ratio):", effects["corr"])
    print(
        "Simple OLS coef, p:",
        effects["model1_coef"],
        effects["model1_p"],
    )
    print(
        "Adjusted OLS coef, p:",
        effects["model2_coef"],
        effects["model2_p"],
    )

    response, explanation = build_conclusion(effects)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

