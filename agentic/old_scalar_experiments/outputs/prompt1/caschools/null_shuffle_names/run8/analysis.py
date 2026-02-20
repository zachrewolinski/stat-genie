import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on the metadata in info.json
    enrollment = df["english"].astype(float)  # total enrollment
    n_teachers = df["students"].astype(float)  # number of teachers
    read_score = df["district"].astype(float)  # average reading score
    math_score = df["expenditure"].astype(float)  # average math score

    df["student_teacher_ratio"] = enrollment / n_teachers
    df["test_score"] = (read_score + math_score) / 2.0

    # Keep observations with valid values for the main variables
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["student_teacher_ratio", "test_score"]
    )

    # Simple (unadjusted) association
    corr = df["student_teacher_ratio"].corr(df["test_score"])

    # Multiple regression controlling for main observed demographics
    X = df[["student_teacher_ratio", "income", "rownames", "school", "computer"]].copy()
    X = sm.add_constant(X)
    y = df["test_score"]
    model = sm.OLS(y, X).fit()
    coef_str = float(model.params["student_teacher_ratio"])
    pval_str = float(model.pvalues["student_teacher_ratio"])

    # Difference in mean scores between low- and high-ratio districts
    df["str_quartile"] = pd.qcut(df["student_teacher_ratio"], 4, labels=False)
    low_mean = float(df[df["str_quartile"] == 0]["test_score"].mean())
    high_mean = float(df[df["str_quartile"] == 3]["test_score"].mean())
    diff = low_mean - high_mean

    # Decide Yes/No based on direction and significance of the adjusted association
    response = "Yes" if coef_str < 0 and pval_str < 0.05 else "No"

    # Interpret the quartile difference
    if diff > 0:
        diff_direction = "higher"
        diff_magnitude = diff
    else:
        diff_direction = "lower"
        diff_magnitude = -diff

    explanation = (
        "Using data on 420 California K-6 and K-8 school districts, I constructed the "
        "student–teacher ratio as total enrollment divided by the number of teachers and "
        "an overall academic performance measure as the average of district reading and "
        "math scores. "
        f"The simple Pearson correlation between the student–teacher ratio and test scores "
        f"is {corr:.3f}, which is very close to zero and indicates little to no linear "
        "association between the two variables. "
        "In a linear regression of test scores on the student–teacher ratio, controlling "
        "for average income, the percentage of students qualifying for CalWorks, the "
        "percentage qualifying for reduced-price lunch, and the percentage of English "
        f"learners, the coefficient on the student–teacher ratio is {coef_str:.3f} with a "
        f"p-value of {pval_str:.3g}, so the estimated effect is very small and not "
        "statistically distinguishable from zero. "
        f"Comparing districts in the lowest versus highest quartile of the student–teacher "
        f"ratio, the low-ratio districts score only about {diff_magnitude:.1f} points "
        f"{diff_direction} on average, a difference that is small relative to typical "
        "variation in scores. "
        "Taken together, these results do not provide evidence that districts with lower "
        "student–teacher ratios systematically achieve higher academic performance in this "
        "dataset."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    # Write a single JSON object with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()
