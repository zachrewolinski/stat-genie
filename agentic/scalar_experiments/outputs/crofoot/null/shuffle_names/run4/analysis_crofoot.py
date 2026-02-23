import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Map columns to interpretable variables based on metadata description.
    # Outcome: 1 if focal group wins, 0 otherwise.
    df["focal_win"] = df["m_focal"].astype(int)

    # Group sizes (number of individuals).
    # According to metadata:
    # - f_other: number of individuals in focal group
    # - win:    number of individuals in other group
    df["focal_group_size"] = df["f_other"]
    df["other_group_size"] = df["win"]
    df["size_adv"] = df["focal_group_size"] - df["other_group_size"]

    # Contest location: distance (m) from each group's home-range center.
    # According to metadata:
    # - m_other: distance of focal group from its home-range center
    # - n_focal: distance of other group from its home-range center
    df["focal_dist"] = df["m_other"]
    df["other_dist"] = df["n_focal"]

    # Location advantage for focal group:
    # positive if the other group is farther from its center than the focal group
    # (i.e. contest is closer to focal group's core area).
    df["loc_adv"] = df["other_dist"] - df["focal_dist"]

    # Fit logistic regression with cluster-robust SEs by dyad to account for
    # repeated contests between the same pair of groups.
    model = smf.glm(
        "focal_win ~ size_adv + loc_adv",
        data=df,
        family=sm.families.Binomial(),
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    params = result.params
    pvalues = result.pvalues

    # Compute effect sizes: change in win probability for a one-unit increase
    # in each predictor at the sample means of the other variables.
    mean_size_adv = df["size_adv"].mean()
    mean_loc_adv = df["loc_adv"].mean()

    def prob(size_adv: float, loc_adv: float) -> float:
        lin = (
            params["Intercept"]
            + params["size_adv"] * size_adv
            + params["loc_adv"] * loc_adv
        )
        return float(1.0 / (1.0 + np.exp(-lin)))

    # One additional individual of size advantage, at average location advantage.
    prob_size_minus = prob(mean_size_adv - 1.0, mean_loc_adv)
    prob_size_plus = prob(mean_size_adv + 1.0, mean_loc_adv)
    delta_size_prob = prob_size_plus - prob_size_minus

    # 100 m shift in location advantage, at average size advantage.
    prob_loc_minus = prob(mean_size_adv, mean_loc_adv - 100.0)
    prob_loc_plus = prob(mean_size_adv, mean_loc_adv + 100.0)
    delta_loc_prob = prob_loc_plus - prob_loc_minus

    # Determine qualitative conclusions based on p-values and effect sizes.
    alpha = 0.05
    size_sig = pvalues["size_adv"] < alpha
    loc_sig = pvalues["loc_adv"] < alpha

    # Build a textual explanation summarizing the evidence.
    explanation_parts = []

    explanation_parts.append(
        "I modeled the probability that the focal capuchin monkey group wins an intergroup contest "
        "using a logistic regression with two predictors: (1) relative group size (focal minus other "
        "group size in number of individuals) and (2) a location advantage index defined as the "
        "difference between the other and focal groups' distances from their respective home-range "
        "centers (positive values indicate the contest is closer to the focal group's core area). "
        "Cluster-robust standard errors by dyad were used to account for multiple contests between "
        "the same pair of groups."
    )

    explanation_parts.append(
        f"The estimated coefficient for relative group size was {params['size_adv']:.3f} "
        f"(p = {pvalues['size_adv']:.3f}). "
        f"This translates into an approximate change in focal win probability of "
        f"{delta_size_prob * 100:.1f} percentage points when the focal group's size advantage "
        f"increases by two individuals (from one individual smaller to one individual larger than "
        f"the opposing group), holding contest location constant. "
        + (
            "This effect is statistically significant at the 5% level, indicating that larger relative group size meaningfully increases the odds of winning."
            if size_sig
            else "This effect is not statistically significant at the 5% level, so the data do not provide strong evidence that relative group size alone reliably predicts contest outcomes."
        )
    )

    explanation_parts.append(
        f"The estimated coefficient for location advantage was {params['loc_adv']:.5f} "
        f"(p = {pvalues['loc_adv']:.3f}). "
        f"A shift of 200 meters in location advantage (from being 100 meters closer to the other "
        f"group's center to being 100 meters closer to the focal group's center) is associated with an "
        f"approximate change in focal win probability of {delta_loc_prob * 100:.1f} percentage points, "
        f"holding group-size advantage constant. "
        + (
            "This effect is statistically significant at the 5% level, supporting the idea that contests occurring closer to a group's home-range center confer a territorial advantage."
            if loc_sig
            else "This effect is not statistically significant at the 5% level, so while the point estimate suggests a possible territorial advantage, the evidence is statistically weak."
        )
    )

    # Overall conclusion and Likert-scale response.
    if size_sig and loc_sig:
        response = 85
        overall_text = (
            "Both relative group size and contest location show statistically reliable associations "
            "with the probability of winning, with larger groups and contests closer to the focal "
            "group's home-range center each increasing the chances of victory. Given the moderate "
            "sample size and effect magnitudes, I interpret this as strong but not absolute evidence "
            "that both factors influence intergroup contest outcomes."
        )
    elif size_sig or loc_sig:
        response = 65
        overall_text = (
            "Only one of the two predictors (either relative group size or contest location) shows "
            "a statistically reliable association with win probability at the 5% level, while the "
            "other shows at best a suggestive but non-significant trend. This provides moderate "
            "evidence that at least one of these factors influences intergroup contest outcomes, but "
            "the joint evidence for both is limited."
        )
    else:
        response = 30
        overall_text = (
            "Neither relative group size nor contest location shows a statistically significant "
            "association with win probability at the 5% level, and effect sizes are modest. "
            "Taken together, the data do not provide strong evidence that these factors reliably "
            "influence intergroup contest outcomes, although small effects cannot be ruled out."
        )

    explanation_parts.append(overall_text)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response), "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

