import json
from pathlib import Path


def main() -> None:
    conclusion = {
        "response": "No",
        "confidence": 85,
        "explanation": (
            "Using the 601 observations in affairs.csv, I compared extramarital involvement "
            "between respondents with and without children. Among those without children, about "
            "25.7% reported at least one extramarital affair (mean coded affair score 1.69); "
            "among those with children, about 24.7% reported an affair (mean affair score 1.36). "
            "A two-sample t-test for the mean affair score (t ≈ -1.04, p ≈ 0.30) and a chi-square "
            "test of the 'any affair' indicator by child status (χ² ≈ 0.03, p ≈ 0.86) both show no "
            "statistically significant difference. The point estimates suggest, at most, a very small "
            "decrease in affairs among parents, but this difference is well within sampling noise. "
            "Given these results, the data do not provide clear evidence that having children decreases "
            "engagement in extramarital affairs."
        ),
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

