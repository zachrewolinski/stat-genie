import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key analytic variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic subset with non-missing ratio and test scores
    df_basic = df.dropna(subset=["stratio", "testscr"]).copy()
    n_basic = len(df_basic)

    # Correlation between student-teacher ratio and test scores
    corr = df_basic["stratio"].corr(df_basic["testscr"])

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(df_basic["stratio"])
    model1 = sm.OLS(df_basic["testscr"], X1).fit()
    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])

    # Multiple regression with available covariates
    candidate_covars = [
        "stratio",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    available_covars = [c for c in candidate_covars if c in df_basic.columns]

    df_multi = df_basic.dropna(subset=available_covars).copy()
    X2 = sm.add_constant(df_multi[available_covars])
    model2 = sm.OLS(df_multi["testscr"], X2).fit()
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])

    # Interpret evidence: lower ratio -> higher performance means
    # testscr decreases as ratio increases (negative association).
    evidence_supports = (
        corr < 0
        and coef1 < 0
        and coef2 < 0
        and pval1 < 0.05
        and pval2 < 0.05
    )
    response = "Yes" if evidence_supports else "No"

    # Confidence calibration based on strength and consistency of evidence
    if evidence_supports:
        if pval1 < 1e-4 and pval2 < 1e-4 and abs(corr) > 0.3:
            confidence = 95
        elif pval1 < 0.01 and pval2 < 0.01 and abs(corr) > 0.2:
            confidence = 90
        else:
            confidence = 80
    else:
        confidence = 60

    # Build explanation using key numerical evidence
    mean_ratio = df_basic["stratio"].mean()
    mean_testscr = df_basic["testscr"].mean()

    if evidence_supports:
        # Narrative for a clear negative association (smaller classes, higher scores)
        explanation = (
            f"We analysed {n_basic} California K-6 and K-8 school districts. "
            f"We defined the student–teacher ratio as students divided by teachers "
            f"(mean {mean_ratio:.1f}) and academic performance as the average of "
            f"5th-grade Stanford 9 reading and math scores (mean {mean_testscr:.1f}). "
            f"The correlation between the student–teacher ratio and test scores was "
            f"{corr:.3f}, indicating that districts with smaller ratios tend to have "
            f"higher scores. In a simple linear regression of test scores on the "
            f"student–teacher ratio, the coefficient on the ratio was {coef1:.3f} "
            f"with p-value {pval1:.3g}. When controlling for income, English-learner "
            f"share, poverty (CalWorks and reduced-price lunch), computers per pupil, "
            f"and expenditures per pupil, the coefficient on the student–teacher ratio "
            f"remained {coef2:.3f} with p-value {pval2:.3g}. These consistent negative "
            f"and statistically significant associations indicate that lower "
            f"student–teacher ratios are associated with higher academic performance "
            f"in this dataset, though the observational nature of the data means we "
            f"cannot make strong causal claims."
        )
    else:
        # Narrative for weak, absent, or opposite association
        direction = "negative" if corr < 0 else "positive"
        explanation = (
            f"We analysed {n_basic} California K-6 and K-8 school districts. "
            f"We defined the student–teacher ratio as students divided by teachers "
            f"(mean {mean_ratio:.1f}) and academic performance as the average of "
            f"5th-grade Stanford 9 reading and math scores (mean {mean_testscr:.1f}). "
            f"The correlation between the student–teacher ratio and test scores was "
            f"{corr:.3f}, a very small {direction} relationship. In a simple linear "
            f"regression of test scores on the student–teacher ratio, the coefficient "
            f"on the ratio was {coef1:.3f} with p-value {pval1:.3g}. When controlling "
            f"for income, English-learner share, poverty (CalWorks and reduced-price "
            f"lunch), computers per pupil, and expenditures per pupil, the coefficient "
            f"on the student–teacher ratio was {coef2:.3f} with p-value {pval2:.3g}. "
            f"These estimates are close to zero and not statistically significant, so "
            f"this dataset does not provide clear evidence that lower student–teacher "
            f"ratios are associated with higher academic performance, and the results "
            f"should be interpreted cautiously given the observational design."
        )

    output = {
        "response": response,
        "confidence": int(round(confidence)),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
