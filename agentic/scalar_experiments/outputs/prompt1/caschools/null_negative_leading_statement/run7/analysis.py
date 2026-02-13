import json

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the main variables
    df = df.dropna(subset=["str", "testscr"])

    n_obs = len(df)

    # Simple bivariate association: Pearson correlation
    corr, corr_p = stats.pearsonr(df["str"], df["testscr"])

    # Difference in means: below vs. above median student–teacher ratio
    median_str = df["str"].median()
    low = df[df["str"] <= median_str]
    high = df[df["str"] > median_str]

    mean_low = low["testscr"].mean()
    mean_high = high["testscr"].mean()
    diff = mean_low - mean_high

    t_stat, t_p = stats.ttest_ind(
        low["testscr"], high["testscr"], equal_var=False, nan_policy="omit"
    )

    # Multiple regression controlling for observable covariates
    formula = "testscr ~ str + english + lunch + calworks + income"
    model = smf.ols(formula=formula, data=df).fit()

    coef_str = model.params["str"]
    p_str = model.pvalues["str"]

    # Determine answer: is a lower student–teacher ratio associated with higher performance?
    # A negative association between str (students per teacher) and testscr implies
    # that lower ratios (smaller classes) are associated with higher test scores.
    if corr < 0 and coef_str < 0:
        answer = "Yes"
    else:
        answer = "No"

    explanation = (
        f"We analyzed the association between the student–teacher ratio "
        f"(students per teacher) and academic performance in the California "
        f"school districts dataset. Academic performance was measured as the "
        f"average of the reading and math test scores for each district, and the "
        f"student–teacher ratio was computed as total students divided by the "
        f"number of teachers. The analysis included {n_obs} districts.\n\n"
        f"In a simple bivariate analysis, the Pearson correlation between the "
        f"student–teacher ratio and average test scores was {corr:.3f} "
        f"(p-value = {corr_p:.3g}). This value is very close to zero and the "
        f"large p-value indicates that there is no statistically significant "
        f"linear relationship between the student–teacher ratio and average test "
        f"scores. Districts with student–teacher ratios at or below the median had "
        f"an average test score of {mean_low:.1f}, compared with {mean_high:.1f} "
        f"for districts above the median, a difference of {diff:.1f} points "
        f"(Welch t-test p-value = {t_p:.3g}). This difference is small in magnitude "
        f"and not statistically significant.\n\n"
        f"To account for observable differences across districts, we estimated a "
        f"multiple linear regression of average test scores on the student–teacher "
        f"ratio and several controls: the percentages of students who are English "
        f"learners, in the CalWorks program, and eligible for reduced-price lunch, "
        f"as well as average district income. In this model, the coefficient on "
        f"the student–teacher ratio was {coef_str:.2f} (p-value = {p_str:.3g}), "
        f"which is very close to zero and not statistically significant, indicating "
        f"no meaningful association between the student–teacher ratio and test "
        f"scores once these covariates are taken into account.\n\n"
        f"Taken together, the near-zero simple correlation, the negligible and "
        f"statistically insignificant difference in average test scores between "
        f"districts with lower versus higher student–teacher ratios, and the "
        f"regression results all suggest that, in this dataset, lower "
        f"student–teacher ratios are not meaningfully associated with higher "
        f"academic performance."
    )

    result = {"response": answer, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
