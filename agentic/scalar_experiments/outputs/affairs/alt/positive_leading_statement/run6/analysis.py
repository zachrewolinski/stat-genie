import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def map_response_from_or(or_est: float, pval: float) -> int:
    """
    Map evidence about whether having children decreases affairs
    onto a 0-100 Likert scale where higher = stronger 'Yes'.
    """
    # Determine direction of association for "children = yes"
    if or_est < 1:
        direction = "decrease"
    elif or_est > 1:
        direction = "increase"
    else:
        direction = "no_change"

    # Basic significance tiers
    if pval < 0.01:
        sig = "strong"
    elif pval < 0.05:
        sig = "moderate"
    elif pval < 0.1:
        sig = "weak"
    else:
        sig = "none"

    if direction == "decrease":
        if sig == "strong":
            response = 85
        elif sig == "moderate":
            response = 75
        elif sig == "weak":
            response = 65
        else:
            response = 55
    elif direction == "increase":
        if sig == "strong":
            response = 15
        elif sig == "moderate":
            response = 25
        elif sig == "weak":
            response = 35
        else:
            response = 45
    else:
        response = 50

    return int(response)


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")
    n = len(df)

    # Binary indicator for having any affairs in the past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_had_affair=("had_affair", "mean"),
            n=("had_affair", "size"),
        )
        .reset_index()
    )
    group_stats = group.set_index("children")

    # Safely extract "yes"/"no" stats if present
    mean_affairs_yes = float(group_stats.loc["yes", "mean_affairs"])
    mean_affairs_no = float(group_stats.loc["no", "mean_affairs"])
    prop_affair_yes = float(group_stats.loc["yes", "prop_had_affair"])
    prop_affair_no = float(group_stats.loc["no", "prop_had_affair"])
    n_yes = int(group_stats.loc["yes", "n"])
    n_no = int(group_stats.loc["no", "n"])

    # Chi-square test for independence between children and any affair
    contingency = pd.crosstab(df["children"], df["had_affair"])
    chi2, chi_p, chi_dof, chi_expected = stats.chi2_contingency(contingency)

    # Logistic regression with controls:
    # had_affair ~ children + age + yearsmarried + religiousness + education
    #             + gender + occupation + rating
    formula = (
        "had_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating"
    )
    model = smf.logit(formula, data=df).fit(disp=False)

    coef_children_yes = model.params["C(children)[T.yes]"]
    pval_children_yes = model.pvalues["C(children)[T.yes]"]
    ci_low, ci_high = model.conf_int().loc["C(children)[T.yes]"]

    or_est = float(np.exp(coef_children_yes))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Map model evidence to 0-100 response
    response = map_response_from_or(or_est, pval_children_yes)

    # Decide qualitative conclusion
    if or_est < 1 and pval_children_yes < 0.05:
        qualitative_conclusion = (
            "there is statistical evidence that having children is associated "
            "with a lower probability of engaging in extramarital affairs, "
            "although the effect size should be interpreted in light of its "
            "magnitude and uncertainty"
        )
    elif or_est < 1 and pval_children_yes >= 0.05:
        qualitative_conclusion = (
            "the point estimates suggest that having children is associated "
            "with slightly fewer extramarital affairs, but this association "
            "is not statistically distinguishable from no effect"
        )
    elif or_est > 1 and pval_children_yes < 0.05:
        qualitative_conclusion = (
            "there is statistical evidence that having children is associated "
            "with a higher probability of engaging in extramarital affairs, "
            "contrary to the hypothesized decrease"
        )
    elif or_est > 1 and pval_children_yes >= 0.05:
        qualitative_conclusion = (
            "the point estimates suggest that having children is associated "
            "with slightly more extramarital affairs, but this association "
            "is not statistically distinguishable from no effect"
        )
    else:
        qualitative_conclusion = (
            "there is no meaningful evidence that having children changes "
            "engagement in extramarital affairs in this dataset"
        )

    # Build explanation string summarizing evidence
    explanation = (
        "Using data on {n} first-married individuals from the Fair (1978) "
        "extramarital affairs study, I examined whether having children is "
        "associated with lower engagement in extramarital affairs. Among "
        "respondents with children (n={n_yes}), the mean affairs score over "
        "the past year was {mean_affairs_yes:.2f}, and "
        "{prop_affair_yes:.1f}% reported at least one affair. Among those "
        "without children (n={n_no}), the mean affairs score was "
        "{mean_affairs_no:.2f}, and {prop_affair_no:.1f}% reported at least "
        "one affair. A chi-square test of the 2x2 table of children status "
        "by any affair yielded chi-square={chi2:.2f} with p-value="
        "{chi_p:.3f}, providing a global test of association. To account for "
        "other factors, I fit a logistic regression model predicting any "
        "affair from children status while adjusting for age, years married, "
        "religiousness, education, gender, occupation, and self-rated "
        "marital happiness. In this model, having children (versus not) had "
        "an odds ratio of {or_est:.2f} for engaging in an extramarital "
        "affair (95% CI [{or_ci_low:.2f}, {or_ci_high:.2f}], p-value="
        "{pval_children_yes:.3f}). Taken together, {qualitative_conclusion}. "
        "The reported 0-100 scalar response reflects the strength and "
        "direction of this evidence, where values above 50 indicate support "
        "for the statement that having children decreases engagement in "
        "extramarital affairs and values below 50 indicate the opposite."
    ).format(
        n=n,
        n_yes=n_yes,
        n_no=n_no,
        mean_affairs_yes=mean_affairs_yes,
        mean_affairs_no=mean_affairs_no,
        prop_affair_yes=prop_affair_yes * 100.0,
        prop_affair_no=prop_affair_no * 100.0,
        chi2=chi2,
        chi_p=chi_p,
        or_est=or_est,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        pval_children_yes=pval_children_yes,
        qualitative_conclusion=qualitative_conclusion,
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

