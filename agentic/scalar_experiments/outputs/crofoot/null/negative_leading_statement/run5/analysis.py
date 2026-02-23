import json
from typing import Any, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_prob_change(
    result: Any,
    X: pd.DataFrame,
    df: pd.DataFrame,
    var: str,
) -> Tuple[float, float, float]:
    """
    Compute the change in predicted probability of win between
    the 25th and 75th percentile of a given predictor, holding
    other predictors at their mean.
    """
    q25, q75 = df[var].quantile([0.25, 0.75])
    base = X.mean()

    low = base.copy()
    high = base.copy()
    low[var] = q25
    high[var] = q75

    low_df = low.to_frame().T
    high_df = high.to_frame().T

    prob_low = float(result.predict(low_df)[0])
    prob_high = float(result.predict(high_df)[0])
    delta = prob_high - prob_low

    return prob_low, prob_high, delta


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct predictors for the research question:
    # relative group size and relative contest location.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]

    y = df["win"]
    X = df[["size_diff", "loc_diff"]]
    X = sm.add_constant(X)

    # Fit logistic regression: probability focal group wins.
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues
    odds_ratios = np.exp(params)

    size_low, size_high, size_delta = compute_prob_change(result, X, df, "size_diff")
    loc_low, loc_high, loc_delta = compute_prob_change(result, X, df, "loc_diff")

    alpha = 0.05
    size_sig = pvalues["size_diff"] < alpha
    loc_sig = pvalues["loc_diff"] < alpha

    # Map statistical evidence to a 0-100 Likert scale where
    # higher numbers indicate stronger "Yes" that the variables
    # influence winning probability.
    if size_sig or loc_sig:
        response = 65
        if size_sig and loc_sig:
            response += 15

        avg_delta = (abs(size_delta) + abs(loc_delta)) / 2.0
        if avg_delta > 0.4:
            response += 10
        elif avg_delta < 0.1:
            response -= 10

        response = max(55, min(95, int(round(response))))
    else:
        response = 20

    explanation_parts = []
    explanation_parts.append(
        "I fit a logistic regression with focal victory (win = 1) as the outcome "
        "and two predictors capturing the study's hypotheses: relative group size "
        "(size_diff = n_focal - n_other) and relative contest location "
        "(loc_diff = dist_other - dist_focal, so positive values mean the contest "
        "is closer to the focal group's home range center)."
    )
    explanation_parts.append(
        f" The estimated coefficients were: size_diff = {params['size_diff']:.3f} "
        f"(p = {pvalues['size_diff']:.3f}, odds ratio = {odds_ratios['size_diff']:.2f}) "
        f"and loc_diff = {params['loc_diff']:.3f} "
        f"(p = {pvalues['loc_diff']:.3f}, odds ratio = {odds_ratios['loc_diff']:.2f})."
    )
    explanation_parts.append(
        f" Moving from the 25th to 75th percentile of size_diff changed the predicted "
        f"probability that the focal group wins from {size_low:.2f} to {size_high:.2f} "
        f"(a change of {size_delta:+.2f}), and for loc_diff from {loc_low:.2f} to "
        f"{loc_high:.2f} (a change of {loc_delta:+.2f})."
    )

    if size_sig or loc_sig:
        explanation_parts.append(
            " At least one of the predictors is statistically significant at the 5% level "
            "and both show nonzero effects on winning probability. "
            "This provides evidence that intergroup contest outcomes do depend on "
            "relative group size and where the contest occurs, contrary to the prior belief "
            "that they do not."
        )
    else:
        explanation_parts.append(
            " Neither predictor is statistically significant at the 5% level, and the implied "
            "changes in winning probability across their observed ranges are modest. "
            "Taken together, the data provide little evidence that relative group size or "
            "contest location meaningfully influence the chance of winning."
        )

    explanation = "".join(explanation_parts)

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
