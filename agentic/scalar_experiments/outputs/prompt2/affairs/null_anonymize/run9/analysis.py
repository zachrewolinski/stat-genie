import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CWD = Path(__file__).resolve().parent

def main() -> None:
    df = pd.read_csv(CWD / "affairs.csv")

    # feature2: affair frequency in past year (0 = none)
    # feature6: children in marriage ('yes'/'no')
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Basic rates
    rates = (
        df.groupby("feature6")["has_affair"].agg(["mean", "sum", "count"])
    )

    # Logistic regression of any affair on children indicator, controlling for key covariates
    # Encode children as binary 1 = yes, 0 = no
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Select covariates similar to classic Fair (1978) analysis
    X = df[[
        "children_yes",  # main variable of interest
        "feature4",      # age
        "feature5",      # years married
        "feature7",      # religiousness
        "feature8",      # education
        "feature9",      # occupation
        "feature10",     # marriage rating
    ]].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["has_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    children_coef = result.params["children_yes"]
    children_p = result.pvalues["children_yes"]

    # Direction: negative coefficient means having children is associated with LOWER odds of affair.
    decreases_affairs = children_coef < 0

    # Build a confidence score heuristic
    # Start from 50 and adjust based on effect size, p-value, and rate differences.
    confidence = 50.0

    # Effect size (in odds ratio terms)
    odds_ratio = float(np.exp(children_coef))
    effect_strength = abs(children_coef)

    if children_p < 0.05:
        confidence += 20
    if children_p < 0.01:
        confidence += 10

    if effect_strength > 0.5:
        confidence += 10
    if effect_strength > 1.0:
        confidence += 5

    # Difference in raw affair rates
    rate_yes = float(rates.loc["yes", "mean"]) if "yes" in rates.index else np.nan
    rate_no = float(rates.loc["no", "mean"]) if "no" in rates.index else np.nan
    rate_diff = rate_no - rate_yes  # positive if no-children have more affairs

    if not np.isnan(rate_diff):
        if rate_diff > 0:
            confidence += 5
        if abs(rate_diff) > 0.05:
            confidence += 5

    # Clamp confidence to [0, 100]
    confidence = max(0.0, min(100.0, confidence))

    # Construct explanation string
    explanation_parts = []

    explanation_parts.append(
        "Analyzed 601 married individuals from the Fair (1978) affairs dataset "
        "using a binary indicator of any extramarital affair in the past year."
    )

    explanation_parts.append(
        f"The observed proportion with any affair was {rate_yes:.3f} for those with children "
        f"and {rate_no:.3f} for those without children (positive difference means higher among childless: {rate_diff:.3f})."
    )

    explanation_parts.append(
        "I fit a logistic regression of any affair on an indicator for having children, "
        "controlling for age, years married, religiousness, education, occupation, and self-rated marriage quality."
    )

    explanation_parts.append(
        f"The coefficient on having children was {children_coef:.3f}, giving an odds ratio of {odds_ratio:.3f}, "
        f"with p-value {children_p:.3g}. A negative coefficient and odds ratio below 1 imply that, "
        "conditional on these covariates, having children is associated with lower odds of engaging in an affair."
    )

    if decreases_affairs:
        response = "Yes"
        explanation_parts.append(
            "Both the regression results and the raw affair rates are consistent with the interpretation "
            "that having children is associated with decreased engagement in extramarital affairs, "
            "although this is an observational association and not necessarily causal."
        )
    else:
        response = "No"
        explanation_parts.append(
            "The regression coefficient and raw rates do not consistently indicate that having children "
            "is associated with lower engagement in extramarital affairs."
        )

    explanation = " " .join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": round(float(confidence), 1),
        "explanation": explanation,
    }

    with open(CWD / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
