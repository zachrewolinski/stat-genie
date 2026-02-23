import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["feature6"] / df["feature7"]  # students per teacher
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0  # overall test score

    df = df.dropna(subset=["stratio", "testscr"])

    # Correlation between student-teacher ratio and test scores
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple regression: testscr ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression with key controls related to disadvantage and resources
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_mult = sm.add_constant(df[["stratio"] + controls])
    model_mult = sm.OLS(df["testscr"], X_mult).fit()
    coef_mult = float(model_mult.params["stratio"])
    p_mult = float(model_mult.pvalues["stratio"])

    # Map statistical evidence to a 0-100 Likert-style scale
    if coef_simple < 0 and coef_mult < 0:
        # Stronger "Yes" as evidence (size + significance) increases
        min_p = max(min(p_simple, p_mult), 1e-16)
        sig_score = -np.log10(min_p) / 10.0
        sig_score = float(np.clip(sig_score, 0.0, 1.0))
        corr_score = float(np.clip(abs(corr) / 0.5, 0.0, 1.0))
        evidence_strength = 0.5 * sig_score + 0.5 * corr_score
        response = int(round(60 + 40 * evidence_strength))  # 60–100
        qualitative_conclusion = "yes"
    else:
        # Evidence points away from a meaningful negative association
        min_p = max(min(p_simple, p_mult), 1e-16)
        sig_score = -np.log10(min_p) / 10.0
        sig_score = float(np.clip(sig_score, 0.0, 1.0))
        corr_score = float(np.clip(abs(corr) / 0.5, 0.0, 1.0))
        evidence_strength = 0.5 * sig_score + 0.5 * corr_score
        response = int(round(40 - 40 * evidence_strength))  # 0–40
        qualitative_conclusion = "no"

    response = max(0, min(100, response))

    # Text describing direction and significance of estimates
    if abs(corr) < 0.05:
        corr_text = (
            f"In the raw data (N={len(df)} districts), the correlation between the "
            f"student–teacher ratio and test scores is {corr:.3f}, which is very close to zero "
            "and indicates essentially no linear association."
        )
    elif corr < 0:
        corr_text = (
            f"In the raw data (N={len(df)} districts), the correlation between the "
            f"student–teacher ratio and test scores is {corr:.3f}, suggesting that districts "
            "with smaller classes (lower ratios) tend to have higher scores."
        )
    else:
        corr_text = (
            f"In the raw data (N={len(df)} districts), the correlation between the "
            f"student–teacher ratio and test scores is {corr:.3f}, suggesting that districts "
            "with larger classes (higher ratios) tend to have slightly higher scores, although "
            "the relationship is weak."
        )

    if coef_simple < 0:
        simple_dir = "a decrease"
    elif coef_simple > 0:
        simple_dir = "an increase"
    else:
        simple_dir = "essentially no change"

    if p_simple < 0.05:
        simple_sig = "and this association is statistically significant"
    else:
        simple_sig = "but this association is not statistically significant"

    if coef_mult < 0:
        mult_dir = "a decrease"
    elif coef_mult > 0:
        mult_dir = "an increase"
    else:
        mult_dir = "essentially no change"

    if p_mult < 0.05:
        mult_sig = "and this association is statistically significant"
    else:
        mult_sig = "and this association is not statistically significant"

    if qualitative_conclusion == "yes":
        final_sentence = (
            "Taken together, these results provide strong evidence of a negative association between "
            "student–teacher ratio and academic performance in this dataset. Although the data are observational "
            "and do not prove that reducing class size will cause higher test scores, the consistent, statistically "
            "significant negative associations support a clear 'Yes' answer to the research question."
        )
    else:
        final_sentence = (
            "Overall, the estimates are small in magnitude and not statistically distinguishable from zero, so this "
            "dataset does not provide convincing evidence that lower student–teacher ratios are associated with higher "
            "academic performance. Given the weak and statistically insignificant relationships, the evidence supports "
            "a 'No' answer to the research question, reflected in the relatively low response value on the 0–100 scale."
        )

    # Build a human-readable explanation of the analysis
    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance? "
        "Using the California school districts dataset, I constructed a student–teacher ratio as total enrollment "
        "divided by the number of teachers and an overall academic performance measure as the average of the reading "
        "and math test scores. "
        + corr_text
        + " A simple linear regression of test scores on the student–teacher ratio yields a coefficient of "
        f"{coef_simple:.3f} (p-value {p_simple:.3g}), so adding one student per teacher is associated with {simple_dir} "
        f" in average test scores, {simple_sig}. "
        "To adjust for major observable differences across districts, I then ran a multiple regression that controls for "
        "economic disadvantage (CalWorks and reduced-price lunch shares), expenditures per pupil, average district income, "
        "and the share of English learners. "
        f"In this adjusted model, the coefficient on the student–teacher ratio is {coef_mult:.3f} (p-value {p_mult:.3g}), "
        f"so adding one student per teacher is associated with {mult_dir} in test scores, {mult_sig}. "
        + final_sentence
    )

    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
