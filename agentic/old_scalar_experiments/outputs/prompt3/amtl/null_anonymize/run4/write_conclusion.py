import json


def main() -> None:
    conclusion = {
        "response": "No",
        "strength": 90,
        "confidence": 90,
        "explanation": (
            "Using binomial regression on the AMTL dataset, modeling the proportion of missing teeth "
            "out of observable sockets with a binomial logit GLM and adjusting for age at death, sex estimate, "
            "and tooth class, humans and non human primates showed nearly identical AMTL frequencies. "
            "After excluding 20 records where the recorded number of missing teeth exceeded the number of observable sockets, "
            "a model with a human indicator versus all non human primates gave a human coefficient close to zero (p ~ 0.92) "
            "and predicted AMTL probabilities of about 5.2% for humans and 5.3% for non human primates at mean age, sex, "
            "and the most common tooth class, and the genus specific model showed Pan slightly higher than Homo sapiens, "
            "Papio similar, and Pongo lower, so there is no evidence that humans have higher AMTL frequencies than non human "
            "primates once these covariates are controlled."
        ),
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

