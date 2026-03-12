import json

response = 30
explanation = (
    "I analyzed 2,362 applications with non-missing gender and decision. "
    "Male applicants had an 88.4% acceptance rate (n=1,867) and female applicants 86.5% (n=495), "
    "a difference of -1.9 percentage points. A two-proportion z-test shows this gap is not statistically "
    "significant (z=-1.16, p=0.245). In a logistic regression with only gender, the female odds ratio "
    "for acceptance is 0.84 (p=0.245). In an adjusted logistic regression controlling for race and "
    "creditworthiness/application factors (black, housing_expense_ratio, self_employed, married, "
    "mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI), the female "
    "odds ratio is 0.85 with a 95% CI of 0.63-1.14 (p=0.269). The effect is small and statistically "
    "indistinguishable from no difference, so the evidence does not support a gender effect on approval."
)

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump({"response": response, "explanation": explanation}, f)
