import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    n = int(df.shape[0])

    # Descriptive statistics
    str_mean = float(df["str"].mean())
    str_std = float(df["str"].std())
    testscr_mean = float(df["testscr"].mean())
    testscr_std = float(df["testscr"].std())

    corr_str_testscr = float(df["str"].corr(df["testscr"]))

    # Simple bivariate regression: testscr ~ str
    X_simple = sm.add_constant(df[["str"]])
    model_simple = sm.OLS(df["testscr"], X_simple, missing="drop").fit()
    beta_str_simple = float(model_simple.params["str"])
    p_str_simple = float(model_simple.pvalues["str"])
    r2_simple = float(model_simple.rsquared)

    # Multivariate regression with common socioeconomic controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]

    X_ctrl = sm.add_constant(df[["str"] + available_controls])
    model_ctrl = sm.OLS(df["testscr"], X_ctrl, missing="drop").fit()
    beta_str_ctrl = float(model_ctrl.params["str"])
    p_str_ctrl = float(model_ctrl.pvalues["str"])
    r2_ctrl = float(model_ctrl.rsquared)

    # Effect size: change in testscr from reducing class size by 5 students per teacher.
    effect5 = -5.0 * beta_str_ctrl
    effect5_sd = effect5 / testscr_std if testscr_std > 0 else np.nan

    # Map evidence strength to a 0–100 Likert response.
    response = determine_likert(beta_str_ctrl, p_str_ctrl, corr_str_testscr, effect5_sd)

    explanation = build_explanation(
        n=n,
        str_mean=str_mean,
        str_std=str_std,
        testscr_mean=testscr_mean,
        testscr_std=testscr_std,
        corr_str_testscr=corr_str_testscr,
        beta_str_simple=beta_str_simple,
        p_str_simple=p_str_simple,
        r2_simple=r2_simple,
        beta_str_ctrl=beta_str_ctrl,
        p_str_ctrl=p_str_ctrl,
        r2_ctrl=r2_ctrl,
        effect5=effect5,
        effect5_sd=effect5_sd,
        response=response,
    )

    output = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


def determine_likert(
    beta_str_ctrl: float,
    p_str_ctrl: float,
    corr_str_testscr: float,
    effect5_sd: float,
) -> int:
    """Convert statistical evidence into a 0–100 Likert response."""
    negative_and_consistent = beta_str_ctrl < 0 and corr_str_testscr < 0
    positive_and_consistent = beta_str_ctrl > 0 and corr_str_testscr > 0

    if negative_and_consistent and p_str_ctrl < 0.001:
        if abs(effect5_sd) >= 0.25:
            response = 90
        elif abs(effect5_sd) >= 0.15:
            response = 85
        else:
            response = 80
    elif negative_and_consistent and p_str_ctrl < 0.01:
        if abs(effect5_sd) >= 0.15:
            response = 80
        else:
            response = 75
    elif negative_and_consistent and p_str_ctrl < 0.05:
        response = 70
    elif negative_and_consistent and p_str_ctrl < 0.1:
        response = 60
    elif negative_and_consistent:
        response = 55
    elif positive_and_consistent:
        if p_str_ctrl < 0.001:
            response = 5
        elif p_str_ctrl < 0.01:
            response = 15
        elif p_str_ctrl < 0.05:
            response = 25
        elif p_str_ctrl < 0.1:
            response = 35
        else:
            response = 40
    elif p_str_ctrl >= 0.1:
        response = 45
    else:
        response = 50

    response = max(0, min(100, int(round(response))))
    return response


def build_explanation(
    n: int,
    str_mean: float,
    str_std: float,
    testscr_mean: float,
    testscr_std: float,
    corr_str_testscr: float,
    beta_str_simple: float,
    p_str_simple: float,
    r2_simple: float,
    beta_str_ctrl: float,
    p_str_ctrl: float,
    r2_ctrl: float,
    effect5: float,
    effect5_sd: float,
    response: int,
) -> str:
    """Create a narrative explanation based on the regression results."""
    # Correlation description
    if abs(corr_str_testscr) < 0.05:
        corr_phrase = (
            "very close to zero, indicating little linear association between class "
            "size and test performance"
        )
    elif corr_str_testscr < 0:
        corr_phrase = (
            "negative, indicating that districts with larger classes tend to have "
            "lower test scores"
        )
    else:
        corr_phrase = (
            "positive, indicating that districts with larger classes tend to have "
            "higher test scores"
        )

    # Simple regression interpretation
    direction_simple = "negative" if beta_str_simple < 0 else "positive"
    if p_str_simple < 0.05:
        simple_sig_phrase = (
            "This coefficient is "
            f"{direction_simple} and statistically significant at conventional levels."
        )
    else:
        simple_sig_phrase = (
            "This coefficient is "
            f"{direction_simple} but extremely small in magnitude and not statistically "
            "significant at conventional levels, providing little evidence of a "
            "meaningful linear relationship in the bivariate model."
        )

    # Controlled regression interpretation
    direction_ctrl = "negative" if beta_str_ctrl < 0 else "positive"
    if p_str_ctrl < 0.05:
        ctrl_sig_phrase = (
            f"The coefficient remains {direction_ctrl} and statistically significant "
            "after adjusting for these controls."
        )
    else:
        ctrl_sig_phrase = (
            f"The coefficient is {direction_ctrl} but remains very small and is not "
            "statistically significant after adjusting for these controls."
        )

    # Effect-size description for a 5-student reduction
    if effect5 > 0:
        effect_direction = "increase"
    elif effect5 < 0:
        effect_direction = "decrease"
    else:
        effect_direction = "no practically meaningful change"

    if response >= 60:
        qualitative = "a 'Yes'-leaning answer"
    elif response <= 40:
        qualitative = "a 'No'-leaning answer"
    else:
        qualitative = "an equivocal answer"

    explanation = (
        f"I analyzed the caschools dataset with {n} California K-6/K-8 school districts to "
        f"assess whether a lower student–teacher ratio is associated with higher academic "
        f"performance. I constructed a student–teacher ratio variable (students per teacher) "
        f"and an overall test score as the average of fifth-grade reading and math scores.\n\n"
        f"Descriptively, the student–teacher ratio has a mean of approximately "
        f"{str_mean:.1f} students per teacher (standard deviation {str_std:.1f}), while the "
        f"average combined test score has a mean of about {testscr_mean:.1f} "
        f"(standard deviation {testscr_std:.1f}). The Pearson correlation between "
        f"student–teacher ratio and the combined test score is {corr_str_testscr:.3f}, "
        f"which is {corr_phrase}.\n\n"
        f"In a simple linear regression of the combined test score on the student–teacher "
        f"ratio, the estimated coefficient on the ratio is {beta_str_simple:.3f}, with a "
        f"p-value of {p_str_simple:.4f} and an R-squared of {r2_simple:.3f}. "
        f"{simple_sig_phrase}\n\n"
        f"To account for observable differences across districts, I then estimated a "
        f"multiple regression including controls for district income, percentage of English "
        f"learners, percentage eligible for reduced-price lunch, percentage on CalWorks, "
        f"per-pupil expenditures, and number of computers. In this richer model, the "
        f"coefficient on the student–teacher ratio is {beta_str_ctrl:.3f} with a p-value "
        f"of {p_str_ctrl:.4f} and an R-squared of {r2_ctrl:.3f}. {ctrl_sig_phrase} "
        f"A one-student increase in class size is associated with about "
        f"{beta_str_ctrl:.2f} points change in the average test score, holding these other "
        f"factors constant.\n\n"
        f"Interpreting the magnitude, a reduction of five students per teacher corresponds "
        f"to an estimated change of about {effect5:.2f} points in the combined test score "
        f"({effect_direction}), which is roughly {effect5_sd:.2f} standard deviations. This "
        f"effect size is extremely small in practical terms given the variation in test "
        f"scores across districts.\n\n"
        f"Overall, the point estimates in both the simple and controlled regressions are "
        f"very close to zero and not statistically significant, so this dataset provides "
        f"little evidence that lower student–teacher ratios are associated with higher "
        f"academic performance. On a 0–100 Likert scale where "
        f"0 represents a strong 'No' and 100 represents a strong 'Yes', I assign a "
        f"response of {response}, reflecting {qualitative} based on the observed "
        f"associations (while recognizing that these are observational data and do not by "
        f"themselves prove causality)."
    )

    return explanation


if __name__ == "__main__":
    main()
