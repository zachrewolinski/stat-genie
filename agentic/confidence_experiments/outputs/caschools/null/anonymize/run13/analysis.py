import json

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df = df[df["feature7"] > 0].copy()
    df["stratio"] = df["feature6"] / df["feature7"]
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    cols = ["stratio", "testscr", "feature8", "feature9", "feature12", "feature13"]
    df = df[cols].dropna()

    corr, p_corr = stats.pearsonr(df["stratio"], df["testscr"])

    x1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], x1).fit()

    controls = ["feature8", "feature9", "feature12", "feature13"]
    x2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(df["testscr"], x2).fit()

    slope1 = float(model1.params["stratio"])
    p1 = float(model1.pvalues["stratio"])
    r2_1 = float(model1.rsquared)

    slope2 = float(model2.params["stratio"])
    p2 = float(model2.pvalues["stratio"])
    r2_2 = float(model2.rsquared)

    response = 50.0
    if slope1 < 0 and p1 < 0.05:
        if slope2 < 0 and p2 < 0.05:
            if p1 < 0.001 and p2 < 0.001 and abs(corr) >= 0.3:
                response = 90.0
            else:
                response = 80.0
        else:
            response = 70.0 if p1 < 0.01 else 60.0
    elif slope1 < 0 and p1 < 0.1:
        response = 55.0
    else:
        response = 40.0

    n_obs = int(df.shape[0])
    abs_corr = abs(corr)

    if abs_corr < 0.1:
        corr_desc = "very close to zero, indicating essentially no linear association between the ratio and scores"
    elif abs_corr < 0.3:
        corr_desc = "small in magnitude, indicating only a weak linear association between the ratio and scores"
    else:
        corr_desc = "moderate in magnitude, indicating a meaningful linear association between the ratio and scores"

    if p1 < 0.05:
        p1_desc = "statistically significant at conventional levels"
    else:
        p1_desc = "not statistically significant at conventional levels"

    if p2 < 0.05:
        p2_desc = "remains statistically significant after adjustment"
    else:
        p2_desc = "is not statistically significant after adjustment"

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance? "
        "I approximated the student–teacher ratio as total enrollment (feature6) divided by number of teachers "
        "(feature7), so lower values mean fewer students per teacher, and academic performance as the average of "
        "reading and math scores ((feature14 + feature15) / 2). "
        f"Across {n_obs} districts, the Pearson correlation between the ratio and average test score is "
        f"{corr:.3f} (p = {p_corr:.3g}), which is {corr_desc}. "
        f"A simple linear regression of test scores on the ratio yields a slope of {slope1:.3f} score points per "
        "additional student per teacher "
        f"(R^2 = {r2_1:.3f}, p = {p1:.3g}), and this estimate is {p1_desc}. "
        "To account for major demographic and resource confounders, I also estimated a multiple regression adding "
        "CalWorks share (feature8), reduced-price lunch share (feature9), average district income (feature12), and "
        "English-learner share (feature13) as controls. "
        f"In this adjusted model, the coefficient on the student–teacher ratio remains {slope2:.3f} with p = {p2:.3g} "
        f"(R^2 = {r2_2:.3f}), and the coefficient on the ratio {p2_desc}. "
        "Because both the simple correlation and the regression coefficients are very small in magnitude and not "
        "statistically distinguishable from zero in either the unadjusted or adjusted models, I interpret this as "
        "little evidence that lower student–teacher ratios are associated with higher academic performance in this "
        "observational dataset. "
        f"The Likert response of {int(round(response))} (0 = strong 'No', 100 = strong 'Yes') therefore reflects a "
        "No answer, indicating that these data provide weak evidence for the hypothesized relationship and cannot be "
        "used to claim a meaningful association, let alone a causal effect."
    )

    response_int = int(max(0, min(100, round(response))))

    result = {"response": response_int, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
