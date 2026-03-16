import json


def main() -> None:
    response = 65
    explanation = (
        "Using 629 children from 8 societies (ages 4 to 14), "
        "I recoded the trial outcome into two variables: overall use of social information "
        "(choosing any demonstrated option versus an undemonstrated option) and preference for "
        "the majority demonstrators among children who followed any demonstrator. I then fitted "
        "binomial logistic regression models with culture (8-level factor) and age (both "
        "continuous and grouped into 4-6, 7-9, 10-12, and 13-14 years) as predictors, and also "
        "examined simple choice proportions.\n\n"
        "First, children showed a high overall reliance on social information: about 79 to 83 "
        "percent of choices followed one of the demonstrated options (only about 21 percent "
        "chose the undemonstrated alternative). Logistic models predicting this social-use "
        "indicator from age and culture showed no statistically reliable effects: age was "
        "non-significant in the linear-age model (p > 0.50), age-group contrasts were also "
        "non-significant (all p >= 0.26), and all culture coefficients were non-significant "
        "(all p >= 0.33). Pseudo R-squared values were very small (around 0.006 to 0.008), "
        "indicating little systematic variation in basic reliance on social information across "
        "cultures or across the 4-14 year age range.\n\n"
        "In contrast, when focusing on preference for the majority among children who did follow "
        "a demonstrator (N = 496), I found clear cross-cultural differences. Overall, 45.6 "
        "percent of all children chose the majority option, 33.2 percent chose the minority "
        "option, and 21.1 percent chose the undemonstrated option. Among social users only, "
        "the majority-choice rate by culture ranged from about 0.41 in culture 1 to about 0.66 "
        "in cultures 5 and 6. In a logistic regression of majority choice on culture and age, "
        "several cultures (3, 4, 5, and 6) had significantly higher odds of choosing the "
        "majority than culture 1 (p values between 0.006 and 0.05), demonstrating robust "
        "cross-cultural variation in majority-bias strength. Age did not predict majority "
        "choice (p > 0.75), and re-running the model with age groups again produced "
        "non-significant age effects (all p >= 0.56) with small pseudo R-squared values "
        "(around 0.02 to 0.024).\n\n"
        "Taken together, these results indicate that children in this study strongly rely on "
        "social information in general, that their preference for majority cues varies "
        "meaningfully across cultural contexts, but that there is little evidence for a strong "
        "developmental trend in these measures between ages 4 and 14. Because the research "
        "question asks whether reliance on social information and majority preference vary "
        "across both cultures and developmental stages, I treat the pattern as partial support: "
        "there is clear cultural modulation of majority preference but only weak, "
        "non-significant evidence for developmental differences. I therefore answer "
        "\"Yes\" overall, but with moderate (not maximal) confidence, which I encode as a "
        "response score of 65 on a 0-100 Likert scale (0 = strong \"No\", 100 = strong \"Yes\")."
    )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

