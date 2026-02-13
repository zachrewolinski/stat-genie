import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0] if info.get("research_questions") else ""

    df = pd.read_csv("caschools.csv")

    # Map scrambled column names to their semantic meaning using info.json descriptions.
    enrollment = df["english"].astype(float)  # total enrollment
    teachers = df["students"].astype(float)  # number of teachers
    read_score = df["district"].astype(float)  # average reading score
    math_score = df["expenditure"].astype(float)  # average math score

    # Construct key variables
    stratio = enrollment / teachers  # students per teacher
    testscr = (read_score + math_score) / 2.0

    print("Research question:", question)
    print(f"Number of districts: {len(df)}")
    print(
        "Student-teacher ratio (students per teacher): "
        f"mean={stratio.mean():.2f}, std={stratio.std():.2f}, "
        f"min={stratio.min():.2f}, max={stratio.max():.2f}"
    )
    print(
        "Average test score (reading/math mean): "
        f"mean={testscr.mean():.2f}, std={testscr.std():.2f}, "
        f"min={testscr.min():.2f}, max={testscr.max():.2f}"
    )

    # Simple correlation
    corr = np.corrcoef(stratio, testscr)[0, 1]
    print(f"\nCorrelation between student-teacher ratio and test scores: {corr:.3f}")

    # Bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(stratio.rename("stratio"))
    model1 = sm.OLS(testscr, X1).fit()
    b1 = model1.params["stratio"]
    se1 = model1.bse["stratio"]
    t1 = model1.tvalues["stratio"]
    p1 = model1.pvalues["stratio"]

    print("\nBivariate OLS: testscr ~ stratio")
    print(
        f"  coef(stratio) = {b1:.3f}, se = {se1:.3f}, "
        f"t = {t1:.2f}, p-value = {p1:.4f}"
    )

    # Multiple regression controlling for key demographics and resources.
    # According to metadata:
    #   income    -> district average income (1,000s USD)
    #   school    -> % eligible for reduced price lunch
    #   computer  -> % CalWorks
    #   rownames  -> % English learners
    #   county    -> % white
    #   grades    -> computers per classroom
    controls = df[["income", "school", "computer", "rownames", "county", "grades"]].astype(
        float
    )
    X2 = sm.add_constant(pd.concat([stratio.rename("stratio"), controls], axis=1))
    model2 = sm.OLS(testscr, X2).fit()
    b2 = model2.params["stratio"]
    se2 = model2.bse["stratio"]
    t2 = model2.tvalues["stratio"]
    p2 = model2.pvalues["stratio"]

    print("\nMultiple OLS: testscr ~ stratio + controls")
    print(
        f"  coef(stratio) = {b2:.3f}, se = {se2:.3f}, "
        f"t = {t2:.2f}, p-value = {p2:.4f}"
    )

    # Also report model fit
    print(f"\nModel fit:")
    print(f"  Bivariate R^2 = {model1.rsquared:.3f}")
    print(f"  Multiple  R^2 = {model2.rsquared:.3f}")

    # Robustness check: trim extreme student-teacher ratios (1st and 99th percentiles)
    lower = np.percentile(stratio, 1)
    upper = np.percentile(stratio, 99)
    mask = (stratio >= lower) & (stratio <= upper)
    str_trim = stratio[mask]
    testscr_trim = testscr[mask]

    corr_trim = np.corrcoef(str_trim, testscr_trim)[0, 1]
    X1_trim = sm.add_constant(str_trim.rename("stratio"))
    model1_trim = sm.OLS(testscr_trim, X1_trim).fit()

    b1t = model1_trim.params["stratio"]
    se1t = model1_trim.bse["stratio"]
    t1t = model1_trim.tvalues["stratio"]
    p1t = model1_trim.pvalues["stratio"]

    print("\nRobustness (trimmed 1st–99th percentile of ratio):")
    print(f"  Correlation (trimmed) = {corr_trim:.3f}")
    print(
        f"  Bivariate coef(stratio) = {b1t:.3f}, se = {se1t:.3f}, "
        f"t = {t1t:.2f}, p-value = {p1t:.4f}"
    )


if __name__ == "__main__":
    main()
