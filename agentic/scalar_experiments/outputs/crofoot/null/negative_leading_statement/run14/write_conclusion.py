import json
from pathlib import Path


def main() -> None:
    response = 25
    explanation = (
        "Using 58 recorded intergroup contests between capuchin monkey groups, "
        "I modeled the probability that the focal group won (win = 1) as a "
        "function of (1) relative group size and (2) contest location. "
        "Relative group size was defined as n_focal − n_other, and contest "
        "location advantage was defined as the difference in distance from each "
        "group’s home-range center (dist_focal − dist_other), so negative values "
        "indicate that the focal group was closer to the center of its own "
        "range than the opponent was to its range center. Both predictors were "
        "standardized before fitting a logistic regression.\n\n"
        "In this logistic model, the standardized relative size coefficient was "
        "−0.30 (odds ratio ≈ 0.74) with p ≈ 0.31, and the standardized location "
        "advantage coefficient was −0.18 (odds ratio ≈ 0.83) with p ≈ 0.53. "
        "Neither effect approaches conventional thresholds for statistical "
        "significance, and a likelihood-ratio test comparing the full model to "
        "a null model with only an intercept produced LR ≈ 1.09 with 2 degrees "
        "of freedom and p ≈ 0.58. This indicates that including relative group "
        "size and contest location does not significantly improve our ability to "
        "predict contest outcomes over a model that assumes a constant win "
        "probability for the focal group.\n\n"
        "Given these results, the available data provide little evidence that "
        "either relative group size or contest location has a reliable, "
        "detectable influence on the probability of winning an intergroup "
        "contest. Because the sample size is modest, we cannot rule out small "
        "or context-dependent effects, but within this dataset any such effects "
        "are too weak to be distinguished from noise. I therefore answer 'No' "
        "to the research question, with moderate strength, reflected by a "
        "Likert-scale score of 25 out of 100 (where 0 is a very strong 'No' and "
        "100 is a very strong 'Yes')."
    )

    obj = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(obj))


if __name__ == "__main__":
    main()

