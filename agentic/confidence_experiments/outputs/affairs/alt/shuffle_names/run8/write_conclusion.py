import json
from pathlib import Path


def main() -> None:
    # Based on the analysis in analyze_affairs.py:
    # - 15.8% of respondents without children reported any extramarital intercourse,
    #   versus 28.6% of those with children.
    # - The unadjusted odds of any affair were about twice as high for those with
    #   children (OR ≈ 2.14, 95% CI ≈ 1.35–3.39, p ≈ 0.001).
    # - After adjusting for age, years married, gender, religiousness, education,
    #   occupation, and marriage rating, the estimated odds ratio remained above 1
    #   (OR ≈ 1.49) but was no longer statistically significant at the 5% level
    #   (95% CI ≈ 0.84–2.64, p ≈ 0.17).
    #
    # Taken together, there is no empirical support in this dataset for the claim
    # that having children *decreases* engagement in extramarital affairs; if
    # anything, the point estimates lean in the opposite direction. To reflect a
    # strong "No" to the research question, while allowing for statistical
    # uncertainty in the adjusted model, we place the answer near the "No" end of
    # the Likert scale.

    response_value = 10  # 0 = strong "No", 100 = strong "Yes"

    explanation = (
        "Using the 601 married respondents in this sample, I coded the outcome as "
        "whether the respondent reported any extramarital sexual intercourse in the "
        "past year (0 = none, >0 = at least once) and the main predictor as whether "
        "there are children in the marriage (yes/no). Among respondents without "
        "children, 27 of 171 (15.8%) reported any extramarital intercourse, whereas "
        "among those with children, 123 of 430 (28.6%) did so; mean affair frequency "
        "was also higher for those with children (≈1.67 vs ≈0.91). A logistic "
        "regression of any affair on children status alone yielded an odds ratio of "
        "about 2.14 for respondents with children compared to those without "
        "(95% CI ≈ 1.35–3.39, p ≈ 0.001), indicating significantly higher—not lower—"
        "odds of reporting an affair for those with children. When I fit an adjusted "
        "logistic model that controlled for age, years married, gender, religiousness "
        "level, education, occupation, and self-rated marriage quality, the estimated "
        "odds ratio for having children was still above 1 (≈1.49), but its 95% "
        "confidence interval included 1 (≈0.84–2.64) and the p-value rose to about "
        "0.17, so the association was no longer statistically significant. Overall, "
        "the descriptive statistics and both models provide no evidence that having "
        "children decreases engagement in extramarital affairs in this dataset; if "
        "anything, the point estimates suggest equal or greater involvement among "
        "parents. Reflecting this, I answer the research question with a strong "
        "\"No\" and place the response near the low end of the 0–100 Likert scale."
    )

    conclusion = {"response": response_value, "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

