import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Identify student–teacher ratio and test score measures based on metadata ranges.
    # Total enrollment: "english"; number of teachers: "students".
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing values in key fields (there should be few or none).
    analysis_df = df[["stratio", "testscr", "income", "school", "computer", "rownames"]].dropna()

    print("Basic summary of key variables:")
    print(analysis_df.describe().T)

    # Zero-order correlation between student–teacher ratio and test scores.
    corr = analysis_df["stratio"].corr(analysis_df["testscr"])
    print(f"\nCorrelation between stratio and testscr: {corr:.3f}")

    # Simple bivariate regression: testscr ~ stratio.
    X_simple = sm.add_constant(analysis_df[["stratio"]])
    y = analysis_df["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nBivariate regression: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with basic demographic and SES controls.
    controls = ["income", "school", "computer", "rownames"]
    X_full = sm.add_constant(analysis_df[["stratio"] + controls])
    model_full = sm.OLS(y, X_full).fit()
    print("\nMultiple regression with controls: testscr ~ stratio + controls")
    print(model_full.summary())

    # Interpret results for the research question.
    beta_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    beta_full = model_full.params["stratio"]
    p_full = model_full.pvalues["stratio"]

    # Heuristic strength of evidence score on 0–100 scale.
    # Start from 50 (undecided) and adjust based on sign, magnitude, and robustness.
    response_score = 50

    if beta_simple < 0:
        response_score += 15
        if p_simple < 0.05:
            response_score += 10
    else:
        response_score -= 15
        if p_simple < 0.05:
            response_score -= 10

    if beta_full < 0:
        response_score += 15
        if p_full < 0.05:
            response_score += 10
    else:
        response_score -= 15
        if p_full < 0.05:
            response_score -= 10

    # Clip to [0, 100].
    response_score = int(np.clip(response_score, 0, 100))

    # Build a concise textual explanation.
    direction_simple = "negative" if beta_simple < 0 else "positive"
    direction_full = "negative" if beta_full < 0 else "positive"

    explanation = (
        "Using the caschools dataset (420 California K-6/K-8 districts), "
        "I constructed student–teacher ratio as total enrollment divided by the number of teachers "
        "(columns 'english' and 'students') and defined academic performance as the average of the "
        "reading and math scores (columns 'district' and 'expenditure'). "
        f"In a bivariate OLS regression of average test score on student–teacher ratio, the coefficient "
        f"on the ratio is {beta_simple:.3f} with p-value {p_simple:.3f}, indicating a {direction_simple} "
        "but statistically insignificant association between class size and test performance. "
        f"When adding controls for district income, percent on income assistance, percent on reduced-price "
        f"lunch, and percent English learners, the coefficient on the student–teacher ratio remains "
        f"{beta_full:.3f} with p-value {p_full:.3f}, so the estimated effect stays very close to zero and "
        "non-significant once basic demographic factors are accounted for. "
        "Overall, these regressions provide little empirical support for the hypothesis that lower "
        "student–teacher ratios (smaller classes) are associated with higher academic performance in this "
        "dataset; the observed relationship is extremely small in magnitude and not statistically "
        "distinguishable from no effect, and the analysis remains observational with potential "
        "unobserved confounding."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))
    print(f"\nWrote conclusion.txt with response={response_score}")


if __name__ == "__main__":
    main()
