import json

# Based on analysis in analysis.py, there is no statistical evidence
# that having children decreases engagement in extramarital affairs.
# Descriptively, parents in this sample have *higher* average affair
# counts and a higher probability of any affair, and regression models
# controlling for age, years married, religiousness, education,
# occupation, and marital rating show a near-zero or slightly positive
# (but not statistically significant) coefficient for having children.

response_value = 10  # 0 = strong "No", 100 = strong "Yes"

explanation = (
    "Using 601 married respondents, I examined whether having children reduces "
    "engagement in extramarital affairs. Descriptively, individuals with children "
    "show a higher mean affair count (1.67 vs 0.91 for those without children) "
    "and a higher proportion reporting any affair in the past year (28.6% vs 15.8%). "
    "To adjust for potential confounders, I fit a Poisson regression for affair "
    "counts and a logistic regression for any affair, both including age, years "
    "married, religiousness, education, occupation, and self-rated marital quality. "
    "In the Poisson model the coefficient for having children is essentially zero "
    "(estimate about 0.00 with a wide 95% confidence interval spanning modest "
    "negative and positive effects and a p-value around 0.97), and in the logistic "
    "model the coefficient is positive but not statistically significant (p ≈ 0.13, "
    "with a confidence interval that includes no effect). Thus, the data provide "
    "no statistical support for the claim that having children decreases "
    "extramarital affairs; if anything, the raw association points in the opposite "
    "direction, and controlled models are consistent with no effect. I therefore "
    "answer 'No' to the research question and place my confidence at 10 on a 0–100 "
    "Likert scale, indicating a strong belief that this dataset does not support a "
    "decreasing effect of children on extramarital affair engagement."
)

with open("conclusion.txt", "w") as f:
    json.dump({"response": response_value, "explanation": explanation}, f)
