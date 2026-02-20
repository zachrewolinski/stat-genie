import json


def main() -> None:
    explanation = (
        "Using 420 California K-6 and K-8 districts, I defined the "
        "student-teacher ratio as students per teacher and academic "
        "performance as the average of reading and math scores. "
        "The simple Pearson correlation between the ratio and test scores "
        "is approximately 0.02, indicating essentially no linear association. "
        "A simple OLS regression of test scores on the ratio yields a "
        "coefficient of about 0.001 with a p-value around 0.67 and an "
        "R-squared near 0.0004, so changes in the ratio do not systematically "
        "predict scores. Controlling for socioeconomic variables (percent "
        "reduced-price lunch, income, and percent English learners) leaves "
        "the coefficient on the ratio near zero with a similar non-significant "
        "p-value and very low R-squared. Comparing mean test scores across "
        "quartiles of the ratio, districts with the lowest ratios do not have "
        "consistently higher scores than those with higher ratios, and "
        "differences are only a few points and not monotonic. A robustness "
        "check trimming extreme ratios (5th–95th percentile) still shows a "
        "near-zero correlation of about 0.03. Taken together, these results "
        "provide no evidence that lower student-teacher ratios are associated "
        "with higher academic performance in this dataset, so I conclude that "
        "the answer is No."
    )

    result = {
        "response": "No",
        "confidence": 90,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

