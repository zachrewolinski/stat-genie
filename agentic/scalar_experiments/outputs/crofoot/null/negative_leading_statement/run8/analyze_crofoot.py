import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Define focal group's relative advantages.
    df["size_advantage"] = df["n_focal"] - df["n_other"]
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]

    X = df[["size_advantage", "loc_advantage"]]
    X = sm.add_constant(X, has_constant="add")
    y = df["win"]

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)

        coeffs = result.params
        pvalues = result.pvalues
        odds_ratios = np.exp(coeffs)

        size_p = float(pvalues["size_advantage"])
        loc_p = float(pvalues["loc_advantage"])

        size_or = float(odds_ratios["size_advantage"])
        loc_or = float(odds_ratios["loc_advantage"])

        pseudo_r2 = float(result.prsquared)

        min_p = min(size_p, loc_p)

        if min_p < 0.01:
            base = 85
        elif min_p < 0.05:
            base = 70
        elif min_p < 0.10:
            base = 55
        else:
            base = 20

        size_log_or = abs(float(np.log(size_or))) if size_or > 0 else 0.0
        loc_log_or = abs(float(np.log(loc_or))) if loc_or > 0 else 0.0
        max_log_or = max(size_log_or, loc_log_or)

        adjustment = 0
        if min_p < 0.05:
            if max_log_or > 0.7:
                adjustment += 10
            elif max_log_or < 0.1:
                adjustment -= 10

        likert = int(round(float(np.clip(base + adjustment, 0, 100))))

        explanation_parts = [
            "I fit a logistic regression predicting whether the focal capuchin group won an intergroup contest (win=1) ",
            "from its size advantage over the opposing group (size_advantage = n_focal - n_other) ",
            "and its location advantage (loc_advantage = dist_other - dist_focal, so positive values mean the focal group ",
            "was closer to the center of its home range than the other group). ",
            f"The model used {len(df)} contests and included both predictors simultaneously. ",
            f"Estimated odds ratios were {size_or:.2f} (p={size_p:.3f}) for size_advantage and ",
            f"{loc_or:.2f} (p={loc_p:.3f}) for loc_advantage, with McFadden pseudo-R² of {pseudo_r2:.3f}. ",
        ]

        if likert > 50:
            explanation_parts.append(
                "Because at least one of these predictors shows statistically meaningful evidence of an association with winning "
                "and the estimated effects are non-trivial in magnitude, I conclude that relative group size and/or contest "
                "location do influence the probability of winning in this dataset. "
            )
        else:
            explanation_parts.append(
                "Both predictors have p-values above conventional significance thresholds and their estimated effects are modest, "
                "so there is little statistical evidence that relative group size or contest location meaningfully influence "
                "the probability of winning in this dataset. "
            )

        explanation_parts.append(
            f"I therefore map my answer onto the 0–100 scale as {likert}, where values near 0 represent a strong 'No' and values near 100 a strong 'Yes'."
        )

        explanation = "".join(explanation_parts)

    except PerfectSeparationError:
        likert = 90
        explanation = (
            "A logistic regression including both size_advantage and loc_advantage failed due to quasi- or perfect separation, "
            "which occurs when combinations of predictors almost perfectly predict contest outcomes. "
            "This provides strong evidence that relative group size and/or contest location influence the probability of winning. "
            "Given this, I assign a high 'Yes' score of 90 on the 0–100 scale, where values near 0 represent a strong 'No' and values near 100 a strong 'Yes'."
        )

    conclusion = {"response": likert, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

