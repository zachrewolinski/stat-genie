import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for ever having an affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children as binary 1/0 (1 = yes, 0 = no)
    df["children_binary"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            std_affairs=("affairs", "std"),
            prop_has_affair=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Two-sample t-test on number of affairs
    affairs_children = df.loc[df["children"] == "yes", "affairs"]
    affairs_no_children = df.loc[df["children"] == "no", "affairs"]
    t_res = stats.ttest_ind(
        affairs_children,
        affairs_no_children,
        equal_var=False,
        nan_policy="omit",
    )

    # Chi-square test on having any affair vs children
    contingency = pd.crosstab(df["children"], df["has_affair"])
    chi2_res = stats.chi2_contingency(contingency)

    # Logistic regression for any affair, controlling for covariates
    # has_affair ~ children + age + yearsmarried + religiousness + education
    #            + occupation + gender + rating
    logit_model = smf.logit(
        "has_affair ~ children_binary + age + yearsmarried + religiousness + "
        "education + C(occupation) + C(gender) + rating",
        data=df,
    ).fit(disp=False)
    logit_params = logit_model.params
    logit_pvalues = logit_model.pvalues

    children_coef = float(logit_params["children_binary"])
    children_p = float(logit_pvalues["children_binary"])
    children_or = float(np.exp(children_coef))

    # Simple Poisson regression on count of affairs (for robustness)
    poisson_model = smf.poisson(
        "affairs ~ children_binary + age + yearsmarried + religiousness + "
        "education + C(occupation) + C(gender) + rating",
        data=df,
    ).fit(disp=False)
    poisson_params = poisson_model.params
    poisson_pvalues = poisson_model.pvalues

    children_coef_pois = float(poisson_params["children_binary"])
    children_p_pois = float(poisson_pvalues["children_binary"])
    children_rr = float(np.exp(children_coef_pois))

    # Derive a scalar response on a 0-100 Likert scale
    # Negative coefficients (odds ratio / rate ratio < 1) indicate
    # lower engagement in extramarital affairs among couples with children.
    evidence_strength = []

    # Use multiple pieces of evidence: logistic, Poisson, t-test, chi-square
    if children_p < 0.05 and children_or < 1:
        evidence_strength.append(80)  # strong evidence from logistic regression
    elif children_p < 0.1 and children_or < 1:
        evidence_strength.append(65)  # moderate evidence from logistic regression
    elif children_p >= 0.1:
        evidence_strength.append(40)  # weak or no evidence in logistic regression

    if children_p_pois < 0.05 and children_rr < 1:
        evidence_strength.append(75)
    elif children_p_pois < 0.1 and children_rr < 1:
        evidence_strength.append(60)
    elif children_p_pois >= 0.1:
        evidence_strength.append(40)

    # t-test on number of affairs
    if t_res.pvalue < 0.05:
        if affairs_children.mean() < affairs_no_children.mean():
            evidence_strength.append(70)
        else:
            evidence_strength.append(30)
    else:
        evidence_strength.append(45)

    # chi-square on any affair vs children
    chi2_p = float(chi2_res[1])
    prop_yes_children = float(group_stats.loc[group_stats["children"] == "yes", "prop_has_affair"])
    prop_yes_no_children = float(group_stats.loc[group_stats["children"] == "no", "prop_has_affair"])
    if chi2_p < 0.05:
        if prop_yes_children < prop_yes_no_children:
            evidence_strength.append(70)
        else:
            evidence_strength.append(30)
    else:
        evidence_strength.append(45)

    # Combine evidence into a single integer score
    response_score = int(round(float(np.mean(evidence_strength))))

    # Clamp to [0, 100]
    response_score = max(0, min(100, response_score))

    # Build explanation text
    explanation_parts = []

    # Summary by group
    explanation_parts.append(
        "We analyzed whether having children is associated with lower engagement "
        "in extramarital affairs using the Fair (1978) marital affairs dataset "
        "(601 married individuals)."
    )

    explanation_parts.append(
        "First, we created a binary outcome 'has_affair' indicating whether a "
        "respondent reported any extramarital intercourse in the past year and "
        "compared both the mean number of affairs and the proportion with at "
        "least one affair between marriages with and without children."
    )

    explanation_parts.append(
        f"In a logistic regression of 'has_affair' on a children indicator "
        f"(1 = children in the marriage) controlling for age, years married, "
        f"religiousness, education, occupation, gender, and self-rated marital "
        f"happiness, the coefficient for having children was "
        f"{children_coef:.3f} (odds ratio {children_or:.3f}, p = {children_p:.3f})."
    )

    explanation_parts.append(
        f"A Poisson regression for the count of affairs with the same controls "
        f"gave a children coefficient of {children_coef_pois:.3f} "
        f"(rate ratio {children_rr:.3f}, p = {children_p_pois:.3f})."
    )

    explanation_parts.append(
        f"A Welch two-sample t-test comparing the mean number of affairs between "
        f"marriages with and without children yielded t = {t_res.statistic:.3f}, "
        f"p = {t_res.pvalue:.3f}. A chi-square test of independence between "
        f"having any affair and having children produced chi-square = "
        f"{chi2_res[0]:.3f}, p = {chi2_p:.3f}."
    )

    explanation_parts.append(
        "Across these models, the point estimates for the children effect were "
        "interpreted in light of their statistical significance. Where p-values "
        "were above conventional thresholds (e.g., 0.05), the data were treated "
        "as providing only weak or inconclusive evidence for any protective effect "
        "of having children on extramarital affairs."
    )

    explanation_parts.append(
        f"Aggregating the strength and direction of evidence from the regression "
        f"models and hypothesis tests yields an overall score of "
        f"{response_score} on a 0–100 scale, where higher values indicate stronger "
        f"support for the claim that having children decreases engagement in "
        f"extramarital affairs."
    )

    if response_score >= 60:
        qualitative = (
            "This corresponds to a 'Yes' answer with at most moderate strength, "
            "reflecting that while some analyses suggest lower affair involvement "
            "among couples with children, the statistical support is limited."
        )
    elif response_score <= 40:
        qualitative = (
            "This corresponds to a 'No' answer or, at best, a very weak 'Yes': "
            "the data do not provide strong or consistent statistical evidence "
            "that having children reduces extramarital affairs."
        )
    else:
        qualitative = (
            "This corresponds to an equivocal answer: the data do not clearly "
            "support or refute a protective effect of having children on "
            "extramarital affairs."
        )

    explanation_parts.append(qualitative)

    explanation = " ".join(explanation_parts)

    # Write required JSON output
    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

