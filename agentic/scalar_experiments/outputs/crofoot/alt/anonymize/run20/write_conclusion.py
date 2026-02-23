import json
from pathlib import Path


def main() -> None:
    # On a 0–100 scale (0 = strong “No”, 100 = strong “Yes”),
    # the analysis supports only a weak, non-significant relationship,
    # so we encode this as a low score leaning toward “No”.
    conclusion = {
        "response": 35,
        "explanation": (
            "Using the 58 recorded intergroup contests, I modelled the probability that the focal capuchin group "
            "won (feature4) as a function of relative group size and contest location. Relative group size was defined "
            "as the difference in total group size between the focal and other group (feature7 − feature8). Contest "
            "location was measured both as the difference in distance from each group to the centre of its home range "
            "(feature5 − feature6) and as a binary indicator for whether the focal group was closer to its own home-"
            "range centre than its opponent. Logistic regressions with win as the response showed a positive but not "
            "statistically significant effect of relative group size (≈0.10 increase in log-odds of winning per extra "
            "individual in the focal group, p≈0.12). The effects of contest location—both the continuous distance "
            "difference and the binary 'focal closer to home' indicator—were small to moderate in magnitude but also "
            "not statistically significant (p-values roughly 0.5 and 0.34, respectively), and a model including size, "
            "location, and their interaction did not yield any predictor with p<0.05. Descriptive comparisons point in "
            "the same qualitative direction: focal groups that are much larger than their opponents (at least three "
            "more individuals) win about 68% of contests versus about 47% otherwise, and focal groups that are closer "
            "to the centre of their home range win about 61% of contests versus about 48% when they are farther away. "
            "However, these contrasts are based on small subgroups (around 20–30 contests per condition), and the "
            "resulting uncertainty is large. Overall, the dataset shows suggestive trends that larger groups and those "
            "fighting closer to their home-range centre may have an advantage, but the evidence is not statistically "
            "robust and could plausibly arise by chance given the limited sample size. Consequently, I interpret the "
            "results as providing insufficient statistical evidence that relative group size and contest location "
            "meaningfully influence win probability in this dataset—a weak 'No' answer—which I encode as a score of "
            "35 on the 0–100 scale (closer to 'No' than to 'Yes')."
        ),
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

