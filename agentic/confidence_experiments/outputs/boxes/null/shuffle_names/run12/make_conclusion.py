import json
from pathlib import Path


def main() -> None:
    explanation = (
        "Using data from 629 children aged 4–14 across eight sites, about 79% "
        "chose one of the demonstrated options (social learning) and 46% copied "
        "the majority demonstrators while 33% copied the minority. Logistic "
        "regression of social-versus-undemonstrated choices on age (centered) "
        "and site indicators showed no reliable developmental or cultural "
        "effects on overall reliance on social information (age p≈0.50; "
        "likelihood-ratio test for site p≈0.87; age-group social-use rates "
        "stay roughly between 0.75 and 0.82 and site-specific rates between "
        "about 0.71 and 0.83). Among children who used social information, a "
        "second logistic model predicting majority-versus-minority copying "
        "again found no evidence that majority preference changes with age "
        "(age p≈0.76), and although some individual sites show higher "
        "majority-bias coefficients, the overall multi-site likelihood-ratio "
        "test does not reach conventional significance (p≈0.12) and observed "
        "majority-copying rates vary only moderately across sites "
        "(roughly 0.33–0.52). Taken together, this pattern suggests that in "
        "this dataset children are consistently social learners with a modest "
        "majority preference that is fairly stable across both developmental "
        "stages and cultural sites, with at most weak, statistically fragile "
        "cross-cultural differences. Therefore I answer that there is not "
        "strong evidence that reliance on social information or preference for "
        "majority cues meaningfully varies across cultures and developmental "
        "stages in this sample."
    )

    conclusion = {
        "response": 30,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

