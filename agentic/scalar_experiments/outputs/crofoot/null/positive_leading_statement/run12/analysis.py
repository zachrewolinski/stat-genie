import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Construct relative group size and relative location advantage.
    # rel_size > 0 means the focal group is larger than the other group.
    df["rel_size"] = df["n_focal"] - df["n_other"]
    # loc_adv > 0 means the focal group is closer to its home-range center
    # than the other group is to its own center.
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    # Center predictors to improve interpretability and reduce collinearity
    df["rel_size_c"] = df["rel_size"] - df["rel_size"].mean()
    df["loc_adv_c"] = df["loc_adv"] - df["loc_adv"].mean()

    # Logistic regression: probability that the focal group wins the contest.
    # Use cluster-robust standard errors by dyad to account for repeated contests
    # between the same pairs of groups.
    model = smf.logit("win ~ rel_size_c + loc_adv_c", data=df)
    res = model.fit(
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": df["dyad"]},
    )

    # Extract coefficients, p-values, and odds ratios from the robust result.
    params = res.params
    pvalues = res.pvalues
    conf_int = res.conf_int()
    odds_ratios = np.exp(params)
    odds_ci = np.exp(conf_int)

    # Pseudo-R^2 from the base model
    pseudo_r2 = float(res.prsquared)

    # Build a compact textual summary used later in the explanation
    lines = []
    lines.append("Logistic regression of win on relative size and location advantage (cluster-robust SE by dyad).")
    for var in ["rel_size_c", "loc_adv_c"]:
        coef = float(params[var])
        pval = float(pvalues[var])
        or_val = float(odds_ratios[var])
        ci_low = float(odds_ci.loc[var, 0])
        ci_high = float(odds_ci.loc[var, 1])
        direction = "positive" if coef > 0 else "negative"
        lines.append(
            f"Predictor {var}: {direction} coefficient = {coef:.3f}, "
            f"odds ratio = {or_val:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f}), p = {pval:.3f}."
        )

    lines.append(f"Model pseudo-R^2 (McFadden) = {pseudo_r2:.3f}.")

    # Interpret the results to construct the Likert-scale response.
    # We consider p < 0.05 as strong evidence, 0.05 <= p < 0.10 as suggestive,
    # and effect size via odds ratios and pseudo-R^2 for strength.
    p_rel = float(pvalues["rel_size_c"])
    p_loc = float(pvalues["loc_adv_c"])
    or_rel = float(odds_ratios["rel_size_c"])
    or_loc = float(odds_ratios["loc_adv_c"])

    # Determine qualitative strength
    significant_rel = p_rel < 0.05
    significant_loc = p_loc < 0.05

    # Base interpretation and scalar
    if significant_rel and significant_loc:
        # Both predictors significantly influence win probability.
        # Use model fit and odds ratios to choose a high score.
        if pseudo_r2 >= 0.3:
            response_score = 90
        elif pseudo_r2 >= 0.15:
            response_score = 80
        else:
            response_score = 70
        qualitative = (
            "Both relative group size and contest location show statistically "
            "significant associations with the probability of the focal group winning."
        )
    elif significant_rel or significant_loc:
        # Only one predictor clearly significant, but the question is about both.
        if pseudo_r2 >= 0.15:
            response_score = 65
        else:
            response_score = 55
        qualitative = (
            "There is clear evidence that at least one of the two factors "
            "(relative group size or contest location) influences win probability, "
            "while the other shows weaker or inconclusive effects."
        )
    else:
        # No strong evidence; answer leans toward "No".
        if pseudo_r2 >= 0.1:
            response_score = 45
        else:
            response_score = 30
        qualitative = (
            "The analysis does not provide strong statistical evidence that "
            "relative group size or contest location reliably influence win probability, "
            "given this dataset."
        )

    # Directional interpretation
    direction_text = (
        "In the fitted model, larger relative group size (focal group being larger "
        "than its opponent) and having the contest occur closer to the focal group's "
        "home-range center are both associated with higher odds of the focal group winning, "
        "as reflected by odds ratios greater than 1 for these predictors." if (or_rel > 1 and or_loc > 1) else
        "In the fitted model, the estimated effects of relative group size and location "
        "on win probability are weaker or mixed in direction."
    )

    explanation = (
        "Research question: Do relative group size and contest location influence the "
        "probability that a capuchin monkey group wins an intergroup contest? "
        "I modeled the binary contest outcome (focal group win vs. loss) using logistic "
        "regression with two key predictors: (1) relative group size (number of individuals "
        "in the focal group minus the number in the other group) and (2) contest location "
        "summarized as the difference between the other group's and focal group's distances "
        "from their respective home-range centers (so positive values mean the contest is "
        "closer to the focal group's center). I included both predictors simultaneously and "
        "used cluster-robust standard errors by dyad to account for repeated contests between "
        "the same pairs of groups. "
        + " "
        + " ".join(lines)
        + " "
        + direction_text
        + " Based on the statistical significance of the coefficients, the direction of the "
        "effects, the magnitude of the odds ratios, and the model's pseudo-R^2, I translated "
        "the overall strength of evidence into a 0–100 scale where 0 is a strong 'No' and "
        "100 is a strong 'Yes' regarding whether these factors influence win probability. "
        + qualitative
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    output_path = Path("conclusion.txt")
    output_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
