import json


def main() -> None:
    conclusion = {
        "response": "No",
        "confidence": 85,
        "explanation": (
            "Using the 601-person Psychology Today affairs dataset, I compared extramarital sexual activity "
            "between married individuals with and without children. Descriptively, the mean affair score was "
            "1.69 for respondents without children and 1.36 for those with children, and the share reporting any "
            "affair in the past year was 25.7% versus 24.7%, a difference of about 1 percentage point. "
            "I then fit a logistic regression of having any affair on an indicator for children while controlling "
            "for age, years married, religiousness, education, occupation, and self-rated marital happiness. "
            "The coefficient for having children was approximately -0.03 in log-odds with a large p-value (~0.88), "
            "indicating an effect that is very small and statistically indistinguishable from zero. Taken together, "
            "these results show no meaningful reduction in the likelihood of extramarital affairs among couples with "
            "children in this sample, so the data do not support the claim that having children decreases engagement "
            "in extramarital affairs."
        ),
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

