import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used below, if any
    model_vars = [
        "avgscore",
        "stratio",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "income",
        "english",
    ]
    data = df[model_vars].dropna()

    # Simple descriptive statistics
    corr = data["stratio"].corr(data["avgscore"])
    stratio_desc = data["stratio"].describe()
    score_desc = data["avgscore"].describe()

    # Regression of academic performance on student-teacher ratio
    y = data["avgscore"]
    X = data[
        [
            "stratio",
            "calworks",
            "lunch",
            "computer",
            "expenditure",
            "income",
            "english",
        ]
    ]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    coef = results.params["stratio"]
    se = results.bse["stratio"]
    tval = results.tvalues["stratio"]
    pval = results.pvalues["stratio"]
    r2 = results.rsquared

    # Map t-statistic for stratio into a 0–100 Likert-style support score
    # for the hypothesis: lower student-teacher ratios improve performance.
    # Negative t-values (evidence that higher ratios reduce performance)
    # should push the score toward 100; positive t-values toward 0.
    support_prob = 1.0 / (1.0 + float(np.exp(tval)))
    response = int(round(100 * support_prob))

    explanation_lines = []
    explanation_lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher "
        "academic performance in California K-6 and K-8 districts?"
    )
    explanation_lines.append(
        "I used the provided caschools dataset (420 districts). "
        "Academic performance was measured as the average of the reading and math "
        "Stanford 9 test scores for 5th graders, and the student-teacher ratio was "
        "computed as total students divided by total teachers in each district."
    )
    explanation_lines.append(
        f"The student-teacher ratio ranges from approximately "
        f"{stratio_desc['min']:.1f} to {stratio_desc['max']:.1f} students per teacher "
        f"(mean {stratio_desc['mean']:.1f}, standard deviation {stratio_desc['std']:.1f}). "
        f"Average test scores range from about {score_desc['min']:.1f} to "
        f"{score_desc['max']:.1f} (mean {score_desc['mean']:.1f}, "
        f"standard deviation {score_desc['std']:.1f}). "
        f"The simple Pearson correlation between student-teacher ratio and average score "
        f"is {corr:.3f}, indicating that districts with more students per teacher tend "
        f"to have slightly lower scores."
    )
    explanation_lines.append(
        "To account for other observable differences between districts, I estimated an "
        "ordinary least squares regression of average test scores on the student-teacher "
        "ratio and several controls: the percentages of students on CalWorks and free/"
        "reduced-price lunch, the number of computers, per-pupil expenditure, average "
        "district income, and the percentage of English learners."
    )
    if coef < 0:
        direction_text = (
            "The coefficient is negative, meaning that, holding the controls fixed, "
            "districts with fewer students per teacher tend to have higher test scores."
        )
    elif coef > 0:
        direction_text = (
            "The coefficient is positive, meaning that, if anything, districts with "
            "more students per teacher tend to have slightly higher scores, although "
            "the estimate is very small."
        )
    else:
        direction_text = (
            "The estimated coefficient is essentially zero, implying no detectable "
            "relationship between the student-teacher ratio and test scores once "
            "controls are included."
        )

    if pval < 0.01:
        significance_text = "This association is statistically very strong."
    elif pval < 0.05:
        significance_text = "This association is statistically significant at conventional levels."
    elif pval < 0.1:
        significance_text = (
            "This association is only marginally statistically significant and should "
            "be interpreted cautiously."
        )
    else:
        significance_text = (
            "However, the t-statistic and large p-value indicate that this association "
            "is not statistically distinguishable from zero in this sample."
        )

    explanation_lines.append(
        f"In this regression, the coefficient on the student-teacher ratio is "
        f"{coef:.3f} (standard error {se:.3f}, t = {tval:.2f}, p-value = {pval:.3g}), "
        f"with an R-squared of {r2:.3f}. {direction_text} {significance_text}"
    )
    explanation_lines.append(
        "I then mapped the t-statistic on the student-teacher ratio into a 0–100 Likert "
        "scale for the strength of a 'Yes' answer to the research question using a "
        "smooth logistic transformation, where values near 50 represent ambiguous "
        "evidence, values near 100 represent strong evidence that lower ratios are "
        "associated with better performance, and values near 0 represent strong "
        "evidence against that claim."
    )
    explanation_lines.append(
        f"The resulting score on this scale is {response} out of 100, indicating "
        f"{'strong' if response >= 75 else 'moderate' if response >= 60 else 'modest' if response > 50 else 'little or no'} "
        "evidence that lower student-teacher ratios are associated with higher academic "
        "performance in this dataset, after adjusting for key demographic and resource "
        "variables."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
