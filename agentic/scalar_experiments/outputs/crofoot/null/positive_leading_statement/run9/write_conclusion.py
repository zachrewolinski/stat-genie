import json


def main() -> None:
    response_value = 15
    explanation = (
        "Using 58 observed intergroup contests between capuchin groups, I modeled the "
        "probability that the focal group won as a function of (1) relative group size, "
        "defined as (n_focal - n_other) / (n_focal + n_other), and (2) contest location, "
        "measured as the difference in distance from each group to the center of its home range "
        "(loc_diff = dist_other - dist_focal, so positive values favor the focal group). "
        "A logistic regression with both predictors showed essentially no explanatory power "
        "(pseudo R² ≈ 0.015, likelihood-ratio test p ≈ 0.55), and neither predictor was "
        "statistically significant (rel_size coefficient ≈ -1.22, p ≈ 0.29; loc_diff coefficient "
        "≈ 0.0009, p ≈ 0.53; both 95% confidence intervals comfortably include zero). "
        "Adding an interaction between relative size and location did not improve model fit "
        "(likelihood-ratio test p ≈ 0.75), again indicating no detectable joint effect. "
        "Simple 2×2 contingency analyses tell the same story: contests where the focal group was "
        "numerically larger actually had a slightly lower win rate (9/20 ≈ 45%) than contests "
        "where it was not larger (23/38 ≈ 61%), but this difference was not statistically "
        "significant (chi-square p ≈ 0.39). Similarly, contests closer to the focal group’s "
        "home range center (focal_home = 1) yielded a win rate of 13/27 (≈ 48%), versus 19/31 "
        "(≈ 61%) when the other group was closer, and this difference was also not significant "
        "(chi-square p ≈ 0.46). Taken together, these results provide no statistically reliable "
        "evidence in this dataset that either relative group size or contest location systematically "
        "influences the probability of the focal capuchin group winning. Therefore, I answer the "
        "research question with a No and place my confidence in that No at 15 on a 0–100 Likert "
        "scale, reflecting a fairly strong lack of evidence for the proposed relationships while "
        "acknowledging the modest sample size and the possibility of more subtle effects that this "
        "analysis could not detect."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response_value, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

