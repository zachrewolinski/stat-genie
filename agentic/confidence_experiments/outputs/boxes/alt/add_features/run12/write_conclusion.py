import json


def main() -> None:
    response_value = 20
    explanation = (
        "The research question asks whether children’s reliance on social information "
        "and their preference for majority cues vary across cultures and developmental stages. "
        "Using the provided dataset (629 observations with outcome y coded as "
        "1 = undemonstrated option, 2 = majority option, 3 = minority option), I derived two "
        "binary measures: (a) reliance on social information (used_social = 1 if y ≠ 1) and "
        "(b) majority preference among those who used social information "
        "(majority_choice = 1 if y = 2, restricted to y ∈ {2, 3}). "
        "I then modeled each outcome as a function of age and culture using logistic regression, "
        "first with main effects (age + C(culture)) and then with age × culture interactions. "
        "For reliance on social information, the main-effects model had a very small pseudo R² "
        "of about 0.013 and a non-significant likelihood-ratio test (LLR p ≈ 0.38). "
        "Including age × culture interactions increased pseudo R² only to about 0.033, and the "
        "overall model was still not statistically significant (LLR p ≈ 0.12). Age itself was not "
        "a significant predictor (p ≈ 0.84), and although one culture dummy reached nominal "
        "significance, the pattern was not robust in the full interaction model. "
        "Descriptively, the probability of using social information was high in all cultures "
        "(roughly 0.67 to 0.88) and across age quartiles (roughly 0.75 to 0.84), indicating a "
        "consistently strong reliance on social information with only modest variation. "
        "For majority preference, logistic models of majority_choice showed pseudo R² values of "
        "about 0.013 (main effects) and 0.032 (with interactions), with non-significant "
        "likelihood-ratio tests (LLR p ≈ 0.35 and 0.12 respectively). Age again was not a "
        "significant predictor (p ≈ 0.91), and culture effects were small and individually "
        "non-significant. Descriptive proportions of majority choices varied only modestly "
        "across cultures (roughly 0.46 to 0.64) and age bands (roughly 0.54 to 0.64). "
        "Taken together, these results indicate that while participants overall show a high "
        "tendency to use social information and a slight preference for majority over minority "
        "demonstrations, there is little statistically robust evidence in this dataset that "
        "either reliance on social information or majority preference varies strongly with age "
        "or across the cultural groups represented. Therefore, I answer the research question "
        "with a \"No\" and place my confidence at 20 on a 0–100 scale, reflecting moderate "
        "confidence that any true age- or culture-related variation, if present, is small and "
        "not clearly detectable with the current sample."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response_value, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

