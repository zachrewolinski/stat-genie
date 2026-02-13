import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used for regression
    cols = [
        "stratio",
        "testscr",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "income",
        "english",
    ]
    df_clean = df.dropna(subset=cols).copy()

    y = df_clean["testscr"]

    # Simple bivariate regression: test score on student-teacher ratio
    X1 = sm.add_constant(df_clean["stratio"])
    model1 = sm.OLS(y, X1).fit()

    # Multivariate regression controlling for demographics and resources
    X2 = df_clean[
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
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()

    # Key statistics
    corr = float(df_clean["stratio"].corr(df_clean["testscr"]))
    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])

    n = int(df_clean.shape[0])
    mean_str = float(df_clean["stratio"].mean())
    sd_str = float(df_clean["stratio"].std())
    mean_test = float(y.mean())
    sd_test = float(y.std())

    # Map evidence strength onto a 0-100 Likert-style scale
    if coef2 < 0 and pval2 < 0.001 and corr < 0:
        response = 90
    elif coef2 < 0 and pval2 < 0.05 and corr < 0:
        response = 75
    elif coef2 < 0 and pval2 < 0.1 and corr < 0:
        response = 65
    elif coef2 < 0:
        response = 55
    else:
        # Little or contrary evidence for a beneficial association
        if coef2 > 0 and pval2 < 0.05:
            response = 20
        else:
            response = 50

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 school districts?\n\n"
        f"I constructed a student–teacher ratio variable as students per teacher and an overall academic "
        f"performance measure as the average of the district-level reading and math scores. The analytic "
        f"sample contains {n} districts. The mean student–teacher ratio is approximately {mean_str:.1f} "
        f"students per teacher (SD {sd_str:.1f}), and the mean combined test score is about "
        f"{mean_test:.1f} (SD {sd_test:.1f}).\n\n"
        f"First, I examined the simple association between student–teacher ratio and test scores. The "
        f"Pearson correlation between the ratio and test scores is {corr:.3f}, indicating that districts "
        f"with lower student–teacher ratios tend to have "
        f"{'higher' if corr < 0 else 'lower' if corr > 0 else 'similar'} scores on average. A simple "
        f"OLS regression of test scores on the ratio yields a slope of {coef1:.3f} points per one-student "
        f"increase in the ratio (p-value {pval1:.3g}). A negative and statistically significant slope "
        f"would imply that adding students per teacher is associated with lower performance.\n\n"
        f"To account for potential confounding by socio-economic and resource factors, I then fit a "
        f"multiple regression of test scores on the student–teacher ratio controlling for the percentages "
        f"of students on income assistance, qualifying for reduced-price lunch, the number of computers, "
        f"per-pupil expenditure, average district income, and the percentage of English learners. In this "
        f"multivariate model, the coefficient on the student–teacher ratio is {coef2:.3f} "
        f"(p-value {pval2:.3g}). The sign and significance of this adjusted coefficient indicate the "
        f"direction and strength of the association between class size and performance after adjusting "
        f"for observed covariates.\n\n"
        f"Based on these results, the overall evidence "
        f"{'strongly ' if response >= 85 else 'moderately ' if response >= 70 else 'weakly ' if response > 50 else 'does not '}supports "
        f"the claim that lower student–teacher ratios are associated with higher academic performance. "
        f"The response score of {response} on a 0–100 scale reflects both the direction of the estimated "
        f"relationship and the statistical strength of the evidence from the multivariate model."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

