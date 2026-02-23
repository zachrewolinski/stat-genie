import json


def main() -> None:
    # Based on the logistic regression analyses in analysis.py:
    # - Neither relative group size (focal group size minus other group size)
    #   nor contest location (difference in distance to each group's home-range
    #   center, scaled per 100 m) showed statistically significant effects on
    #   the probability that the focal group won.
    # - In a model win ~ rel_group_size + home_adv_100, both predictors had
    #   p-values well above 0.05 (approximately 0.31 and 0.53) and the overall
    #   likelihood-ratio test had p ~ 0.58 with pseudo R^2 ~ 0.014, indicating
    #   very limited explanatory power.
    # - Univariate logistic models for each predictor, and for binary indicators
    #   of "focal group larger" and "focal group closer to home," also yielded
    #   non-significant effects (p-values > 0.25 in all cases).
    # - Descriptively, win rates were actually slightly lower when the focal
    #   group was larger (about 45% vs ~61% when not larger) and when it was
    #   closer to its home-range center (~48% vs ~61% when not closer), which
    #   runs counter to the hypothesized advantages and suggests no clear
    #   positive relationship in this sample of 58 contests.
    #
    # Together, these results provide little evidence that relative group size
    # or contest location meaningfully influence win probability in this dataset,
    # though the modest sample size means small to moderate effects cannot be
    # completely ruled out. Overall, the balance of evidence points toward a
    # "No" answer with reasonably high confidence.

    response_value = 20  # 0=strong "No", 100=strong "Yes"

    explanation = (
        "Using 58 intergroup contests with a binary outcome for whether the focal "
        "capuchin group won (feature4), I modeled win probability as a function of "
        "relative group size and contest location. Relative group size was defined "
        "as focal group size minus other group size (feature7 − feature8), and "
        "contest location was summarized as a home-field advantage variable equal "
        "to the other group’s distance from its home-range center minus the focal "
        "group’s distance, measured per 100 m (home_adv_100 = (feature6 − feature5) / 100). "
        "A logistic regression win ~ rel_group_size + home_adv_100 yielded very small "
        "pseudo R² (~0.014) and a non-significant likelihood-ratio test (p ≈ 0.58), "
        "indicating that these predictors do not jointly explain contest outcomes. "
        "Individually, neither rel_group_size nor home_adv_100 was statistically "
        "significant (p-values ≈ 0.31 and 0.53), and separate univariate logistic "
        "models for each predictor and for binary indicators of 'focal group larger' "
        "and 'focal group closer to home' all had p-values above 0.25. "
        "Simple win-rate comparisons show similar patterns: the focal group won in "
        "about 45% of contests when it was larger versus roughly 61% when it was "
        "not larger, and about 48% of contests when it was closer to its home-range "
        "center versus roughly 61% when it was not, which runs counter to the "
        "expected advantages of size and home location. Taken together, these "
        "results provide little evidence that relative group size or contest "
        "location materially influence the probability of winning in this dataset, "
        "though with only 58 contests small effects cannot be fully ruled out. "
        "Overall, I interpret the data as supporting a 'No' answer, corresponding "
        "to a relatively low Likert-scale value of 20 on a 0–100 scale."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response_value, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

