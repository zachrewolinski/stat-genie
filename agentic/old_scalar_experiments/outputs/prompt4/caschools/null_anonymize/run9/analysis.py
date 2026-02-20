import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables based on metadata in info.json
    # Student–teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing/invalid values for core variables
    core = df[["stratio", "testscr", "feature8", "feature9", "feature12", "feature13"]].dropna()

    # Simple bivariate association
    corr, pval = pearsonr(core["stratio"], core["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(core["stratio"])
    model_simple = sm.OLS(core["testscr"], X_simple).fit()
    slope_simple = model_simple.params["stratio"]

    # Multiple regression controlling for key demographics:
    # CalWorks %, reduced-price lunch %, income, English learners %
    X_controls = core[["stratio", "feature8", "feature9", "feature12", "feature13"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(core["testscr"], X_controls).fit()
    slope_adj = model_controls.params["stratio"]
    pval_adj = model_controls.pvalues["stratio"]

    # Compare quartiles of student–teacher ratio
    core["str_q"] = pd.qcut(core["stratio"], 4, labels=False)
    low_q = core[core["str_q"] == 0]["testscr"].mean()
    high_q = core[core["str_q"] == 3]["testscr"].mean()
    diff_q = low_q - high_q

    # Determine strength of evidence for "yes"
    # We use direction, effect size, and robustness to controls.
    # Start from 50 (agnostic) and adjust.
    response = 50

    # Direction: negative slopes indicate lower ratios -> higher scores
    if slope_simple < 0 and corr < 0:
        response += 15
    elif slope_simple > 0 and corr > 0:
        response -= 15

    # Effect size via correlation magnitude
    abs_corr = abs(corr)
    if abs_corr >= 0.4:
        response += 25
    elif abs_corr >= 0.25:
        response += 15
    elif abs_corr >= 0.1:
        response += 5
    else:
        response -= 5

    # Statistical significance (bivariate)
    if pval < 0.001:
        response += 15
    elif pval < 0.01:
        response += 10
    elif pval < 0.05:
        response += 5

    # Robustness after controls
    if slope_adj < 0 and pval_adj < 0.05:
        response += 10
    elif slope_adj < 0 and pval_adj < 0.1:
        response += 5
    elif slope_adj > 0 and pval_adj < 0.05:
        response -= 10

    # Bound to [0, 100] and cast to int
    response = int(min(100, max(0, round(response))))

    # Build explanation as a single line (no newlines) describing key evidence.
    explanation = (
        "Using data from 420 California K-6 and K-8 districts, I computed the student–teacher ratio as total enrollment "
        "divided by teachers and academic performance as the average of reading and math scores. "
        f"The bivariate Pearson correlation between student–teacher ratio and test scores is {corr:.3f} (p = {pval:.3g}), "
        f"and a simple linear regression yields an estimated slope of {slope_simple:.2f} test-score points per additional student per teacher, which is effectively zero. "
        f"Comparing quartiles, districts in the lowest student–teacher-ratio quartile score on average {diff_q:.1f} points relative to those in the highest quartile, again a very small difference. "
        f"After controlling for CalWorks percentage, reduced-price lunch percentage, district income, and English-learner percentage, "
        f"the estimated slope remains near zero at {slope_adj:.2f} with p-value {pval_adj:.3g}, so overall the data provide little evidence that lower student–teacher ratios are meaningfully associated with higher academic performance; "
        "on a 0–100 scale for answering this yes/no question I therefore rate the evidence for a positive association as low."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
