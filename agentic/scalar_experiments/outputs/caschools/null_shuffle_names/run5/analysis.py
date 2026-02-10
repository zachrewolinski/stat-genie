import math

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Map metadata to semantic variables:
    # - "english" column: total enrollment
    # - "students" column: number of teachers
    # - "district": average reading score
    # - "expenditure": average math score
    enrollment = df["english"]
    teachers = df["students"]

    # Avoid division by zero just in case
    stratio = enrollment / teachers.replace(0, pd.NA)
    stratio = stratio.astype(float)

    # Academic performance: mean of reading and math scores
    testscr = df[["district", "expenditure"]].mean(axis=1)

    # Drop any rows with missing values for a clean analysis
    mask = stratio.notna() & testscr.notna()
    stratio_clean = stratio[mask]
    testscr_clean = testscr[mask]

    # Compute Pearson correlation
    corr = stratio_clean.corr(testscr_clean)

    # Simple OLS: testscr ~ stratio
    X = sm.add_constant(pd.DataFrame({"stratio": stratio_clean}))
    model = sm.OLS(testscr_clean, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]

    # Print summary stats for transparency in the shell
    print("Correlation between STR and test score:", corr)
    print("OLS coefficient on STR:", coef)
    print("p-value for STR coefficient:", pval)

    # Map evidence to Likert scalar (-100 to 100)
    # Research question: "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # If the coefficient on STR is negative (more students per teacher -> lower scores)
    # and statistically significant, that is evidence for a "Yes" answer.
    # We scale strength mainly by |corr| and significance level.
    abs_corr = abs(corr) if not math.isnan(corr) else 0.0
    corr_scale = min(abs_corr / 0.5, 1.0)  # 0.5 or more treated as very strong

    scalar: int
    if coef < 0 and pval < 0.001:
        # Strong evidence in hypothesized (beneficial) direction
        strength = 60 + int(round(40 * corr_scale))
        scalar = max(1, min(100, strength))
    elif coef < 0 and pval < 0.05:
        # Moderate evidence in hypothesized direction
        strength = 30 + int(round(40 * corr_scale))
        scalar = max(1, min(80, strength))
    elif coef < 0:
        # Weak, non-significant evidence in hypothesized direction
        strength = 10 + int(round(20 * corr_scale))
        scalar = max(1, min(40, strength))
    elif coef > 0 and pval < 0.05:
        # Significant evidence in the opposite direction
        strength = 30 + int(round(40 * corr_scale))
        scalar = -max(1, min(80, strength))
    elif coef > 0:
        # Weak evidence opposite to hypothesized direction
        strength = 10 + int(round(20 * corr_scale))
        scalar = -max(1, min(40, strength))
    else:
        # No clear association
        scalar = 0

    # Write scalar conclusion to file with no extra text
    with open("conclusion.txt", "w") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

