import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obvious missing values, if present
    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate regression: testscr on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = sm.add_constant(df[["stratio"] + available_controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()

    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]

    coef_controls = model_controls.params["stratio"]
    p_controls = model_controls.pvalues["stratio"]

    # Correlation for additional descriptive evidence
    corr = df["stratio"].corr(df["testscr"])

    # Translate statistical evidence into a 0–100 confidence score
    # Heuristic:
    # - Strong, consistent negative association with p < 0.01 -> ~85
    # - Moderate evidence (negative, p between 0.01 and 0.05) -> ~70
    # - Weak/non-robust evidence -> <= 55
    response_score: int
    if coef_simple < 0 and coef_controls < 0 and p_simple < 0.01 and p_controls < 0.01:
        response_score = 85
    elif coef_simple < 0 and coef_controls < 0 and p_simple < 0.05 and p_controls < 0.05:
        response_score = 70
    elif coef_simple < 0 and coef_controls < 0 and p_simple < 0.1 and p_controls < 0.1:
        response_score = 60
    elif coef_simple < 0 and coef_controls < 0:
        response_score = 55
    else:
        response_score = 40

    # Construct a concise explanation using key numerical results
    explanation = (
        "I examined whether lower student–teacher ratios are associated with higher academic performance "
        "using the California school districts data. I defined the student–teacher ratio as students divided "
        "by teachers and the test score as the average of reading and math scores. In a simple OLS regression "
        "of average test scores on the student–teacher ratio, the coefficient on the ratio was "
        f"{coef_simple:.2f} with a p-value of {p_simple:.3f}, indicating that higher ratios (more students per "
        "teacher) are associated with "
        f"{'lower' if coef_simple < 0 else 'higher'} test scores. The correlation between the ratio and test "
        f"scores was {corr:.2f}. When I added controls for income, poverty and disadvantage (CalWorks, lunch, "
        "English learners), computers per classroom, and spending per student, the coefficient on the "
        f"student–teacher ratio remained {coef_controls:.2f} with a p-value of {p_controls:.3f}. Overall, the "
        "association between lower student–teacher ratios and higher academic performance is "
        f"{'statistically significant and fairly robust' if response_score >= 70 else 'present but only moderately strong'}; "
        "based on this evidence I assign a response of "
        f"{response_score} on a 0–100 scale, where higher values indicate stronger support for the claim that "
        "lower student–teacher ratios are associated with higher academic performance."
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

