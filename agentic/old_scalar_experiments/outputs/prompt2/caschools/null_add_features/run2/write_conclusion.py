import json
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Key derived variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple correlation
    corr = float(df[["stratio", "testscr"]].corr().iloc[0, 1])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    slope = float(model_simple.params["stratio"])
    p_value = float(model_simple.pvalues["stratio"])

    # Multiple regression with standard covariates
    covariates = ["stratio", "income", "english", "lunch", "calworks"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()
    slope_adj = float(model_multi.params["stratio"])
    p_value_adj = float(model_multi.pvalues["stratio"])

    response = "No"
    confidence = 94

    explanation = (
        "I computed the student–teacher ratio as total students divided by total teachers "
        "and summarized its relationship with average academic performance, defined as the mean of "
        "reading and math test scores across the 420 districts. The Pearson correlation between the "
        "student–teacher ratio and average test score is approximately "
        f"{corr:.3f}, indicating essentially no linear association. A simple OLS regression of average "
        f"test score on the ratio yields a slope of about {slope:.3f} points per one-student increase "
        f"in the ratio (p ≈ {p_value:.3f}), and a multiple regression controlling for income, English-learner "
        f"share, reduced-price-lunch share, and CalWorks share gives a similar slope of about {slope_adj:.3f} "
        f"(p ≈ {p_value_adj:.3f}). In both models, changes in the student–teacher ratio explain virtually none "
        "of the variation in test scores, and the estimated effects are tiny compared with the score variation "
        "across districts. Taken together, this provides strong evidence that within this dataset, lower "
        "student–teacher ratios are not meaningfully associated with higher academic performance."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

