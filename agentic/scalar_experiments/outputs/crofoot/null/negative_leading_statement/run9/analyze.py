import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("crofoot.csv")

    # Relative group size: focal group size minus other group size
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Location advantage: how much closer the focal group is to its home range center
    # Positive values mean the contest is closer to the focal group's home range center.
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors to aid interpretability of coefficients
    for col in ["rel_size", "loc_adv"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0:
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Prepare data for logistic regression
    X = df[["rel_size_z", "loc_adv_z"]]
    X = sm.add_constant(X, has_constant="add")
    y = df["win"]

    # Fit logistic regression model
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        converged = result.mle_retvals.get("converged", True)
    except Exception:
        # Fallback: no valid model fit
        result = None
        converged = False

    n_obs = int(len(df))

    response_score = 50  # neutral default
    explanation_parts = []

    if result is not None and converged:
        params = result.params
        pvalues = result.pvalues

        # Extract key statistics
        coef_rel = float(params.get("rel_size_z", np.nan))
        p_rel = float(pvalues.get("rel_size_z", np.nan))

        coef_loc = float(params.get("loc_adv_z", np.nan))
        p_loc = float(pvalues.get("loc_adv_z", np.nan))

        pseudo_r2 = float(result.prsquared)

        # Odds ratios per 1 SD change
        or_rel = math.exp(coef_rel) if not math.isnan(coef_rel) else float("nan")
        or_loc = math.exp(coef_loc) if not math.isnan(coef_loc) else float("nan")

        # Determine strength of evidence based on p-values
        sig_rel = p_rel < 0.05
        sig_loc = p_loc < 0.05
        any_sig = sig_rel or sig_loc
        both_sig = sig_rel and sig_loc

        if both_sig and pseudo_r2 >= 0.2:
            response_score = 85
        elif any_sig and pseudo_r2 >= 0.1:
            response_score = 70
        elif any_sig:
            response_score = 60
        elif (p_rel < 0.1) or (p_loc < 0.1):
            response_score = 55
        else:
            response_score = 25

        # Build explanation
        explanation_parts.append(
            f"I analysed {n_obs} intergroup contests using a logistic regression "
            f"with win (1=focal group won, 0=other group won) as the outcome and "
            f"two predictors: relative group size (focal minus other group size, "
            f"standardised) and contest location advantage (how much closer the focal "
            f"group was to its home range centre compared to the other group, standardised)."
        )

        explanation_parts.append(
            "The model included an intercept and was fitted with maximum likelihood."
        )

        explanation_parts.append(
            f"For relative group size, the estimated coefficient per 1 SD increase "
            f"was {coef_rel:.3f}, corresponding to an odds ratio of {or_rel:.2f}, "
            f"with p-value {p_rel:.3f}."
        )

        explanation_parts.append(
            f"For contest location advantage, the estimated coefficient per 1 SD increase "
            f"was {coef_loc:.3f}, corresponding to an odds ratio of {or_loc:.2f}, "
            f"with p-value {p_loc:.3f}."
        )

        explanation_parts.append(
            f"The model's McFadden pseudo-R² was {pseudo_r2:.3f}, indicating the "
            f"predictors together explain a non-trivial portion of variation in win probability."
        )

        if both_sig:
            explanation_parts.append(
                "Both predictors were statistically significant at the 0.05 level, "
                "providing strong evidence that relative group size and contest location "
                "each influence the probability that a capuchin group wins an intergroup contest."
            )
        elif any_sig:
            explanation_parts.append(
                "At least one of the two predictors was statistically significant at the 0.05 level, "
                "indicating that relative group size and/or contest location influence the "
                "probability that a capuchin group wins an intergroup contest."
            )
        else:
            explanation_parts.append(
                "Neither predictor reached conventional significance at the 0.05 level, so the "
                "evidence that relative group size and contest location influence win probability "
                "is weak in this sample."
            )

        if response_score >= 50:
            explanation_parts.append(
                f"Based on these results, I answer 'Yes' to the research question and place my "
                f"confidence at {response_score} on a 0–100 scale, where higher values represent "
                f"stronger evidence that these factors influence win probability."
            )
        else:
            explanation_parts.append(
                f"Based on these results, I answer 'No' to the research question and place my "
                f"confidence at {response_score} on a 0–100 scale, where lower values represent "
                f"stronger evidence that these factors do not meaningfully influence win probability."
            )

    else:
        # Model did not converge; fall back to simple correlation analysis
        corr_rel = float(df["win"].corr(df["rel_size"]))
        corr_loc = float(df["win"].corr(df["loc_adv"]))

        # Use a conservative response in the absence of a proper model
        response_score = 40 if (abs(corr_rel) > 0.2 or abs(corr_loc) > 0.2) else 30

        explanation_parts.append(
            f"A logistic regression model of win on relative group size and contest "
            f"location failed to converge reliably on this dataset of {n_obs} contests, "
            f"so I instead examined simple Pearson correlations."
        )
        explanation_parts.append(
            f"The correlation between win and relative group size was {corr_rel:.3f}, and "
            f"between win and location advantage was {corr_loc:.3f}."
        )

        if abs(corr_rel) > 0.2 or abs(corr_loc) > 0.2:
            explanation_parts.append(
                "These correlations suggest at least modest associations, but without a stable "
                "regression model the strength and significance of these effects are uncertain."
            )
        else:
            explanation_parts.append(
                "These correlations are small in magnitude, providing little evidence that "
                "relative group size or contest location meaningfully influence win probability."
            )

        explanation_parts.append(
            f"Given these limitations, I give a cautious response score of {response_score} on "
            f"a 0–100 scale."
        )

    explanation = " ".join(explanation_parts)

    # Ensure integer in [0, 100]
    response_int = int(round(max(0, min(100, response_score))))

    output = {"response": response_int, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

