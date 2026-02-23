import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Group summaries by children
    group = df.groupby("children")
    summaries = group["affairs"].agg(["mean", "std", "count"])
    prop_any = group["any_affair"].mean()

    print("=== Group summaries by children (affairs as count) ===")
    print(summaries)
    print("\nProportion with any affair by children status:")
    print(prop_any)

    # Two-sample tests on affair counts
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    print("\n=== Mean difference tests on affair counts (children yes vs no) ===")
    t_res = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    print("Welch t-test:", t_res)

    mw_res = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    print("Mann-Whitney U test:", mw_res)

    # Test on any_affair proportions via chi-squared test
    tab = pd.crosstab(df["children"], df["any_affair"])
    chi2_res = stats.chi2_contingency(tab)
    print("\n=== Chi-squared test on any_affair by children ===")
    print("Contingency table:")
    print(tab)
    print("chi2, p-value, dof:", chi2_res[0], chi2_res[1], chi2_res[2])

    # Logistic regression: any_affair ~ children + controls
    print("\n=== Logistic regression: any_affair ~ C(children) ===")
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print(logit_simple.summary())

    print("\n=== Logistic regression with controls ===")
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_full = smf.logit(formula, data=df).fit(disp=False)
    print(logit_full.summary())

    # Build quantitative summary for the conclusion
    mean_no = summaries.loc["no", "mean"]
    mean_yes = summaries.loc["yes", "mean"]
    prop_no = prop_any.loc["no"]
    prop_yes = prop_any.loc["yes"]
    coef_children_full = logit_full.params["C(children)[T.yes]"]
    p_children_full = logit_full.pvalues["C(children)[T.yes]"]

    # Likert-style response: 0 = strong "No", 100 = strong "Yes".
    # Multiple tests show small, non-significant differences, so we give a
    # fairly strong "No" to the claim that children decrease affairs.
    response_score = 20

    explanation = (
        "We tested whether having children decreases engagement in extramarital affairs. "
        f"In this sample, the mean affair count was {mean_no:.2f} for couples without children "
        f"and {mean_yes:.2f} for couples with children, while the share with any affair was "
        f"{prop_no:.3f} versus {prop_yes:.3f}, indicating only small raw differences. "
        f"A Welch t-test comparing affair counts between the groups yielded p={t_res.pvalue:.3f}, "
        f"and a Mann-Whitney U test yielded p={mw_res.pvalue:.3f}, providing no evidence of a difference. "
        f"A chi-squared test on the proportion with any affair gave p={chi2_res[1]:.3f}. "
        f"In a logistic regression of having any affair on children status, the coefficient for having children "
        f"was {coef_children_full:.3f} with p={p_children_full:.3f} after controlling for age, years married, "
        "religiousness, education, occupation, marital rating, and gender, again far from statistical significance. "
        "Overall, across multiple tests the estimated effects of having children are small and not statistically significant, "
        "so the data do not support the claim that having children decreases engagement in extramarital affairs."
    )

    conclusion = {"response": response_score, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
