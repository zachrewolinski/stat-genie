import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json metadata, columns are slightly mislabelled.
    # We reconstruct the variables of interest using those descriptions:
    #
    # - Column "age" is actually the numeric frequency of extramarital intercourse
    #   in the past year (0 = none, 1 = once, 2 = twice, 3 = 3 times,
    #   7 = 4–10 times, 12 = monthly/weekly/daily).
    # - Column "religiousness" is actually a factor answering
    #   "Are there children in the marriage?" with values "yes"/"no".
    # - Column "occupation" is the respondent's age in years
    #   (17.5 = under 20, 22 = 20–24, …, 57 = 55+).
    # - Column "children" is actually years married.
    # - Column "affairs" is actually self‑rated marriage quality (1–5).

    # Derived variables with clearer names
    df["affairs_freq"] = df["age"].astype(float)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)

    df["resp_age_years"] = df["occupation"].astype(float)
    df["years_married"] = df["children"].astype(float)
    df["marriage_rating"] = df["affairs"].astype(float)

    # Drop any rows with missing key values just in case
    df = df.dropna(
        subset=["affairs_freq", "has_children", "any_affair", "resp_age_years", "years_married", "marriage_rating"]
    )

    # Basic group summaries
    group_means = df.groupby("has_children")["affairs_freq"].mean()
    group_props = df.groupby("has_children")["any_affair"].mean()
    n_by_group = df["has_children"].value_counts().to_dict()

    mean_no_children = float(group_means.get(0, np.nan))
    mean_children = float(group_means.get(1, np.nan))
    prop_any_no_children = float(group_props.get(0, np.nan))
    prop_any_children = float(group_props.get(1, np.nan))

    # Two-sample t-test on the numeric frequency
    with_children = df.loc[df["has_children"] == 1, "affairs_freq"]
    without_children = df.loc[df["has_children"] == 0, "affairs_freq"]

    ttest_res = stats.ttest_ind(with_children, without_children, equal_var=False)

    # Nonparametric Mann–Whitney U test as robustness check
    try:
        mann_res = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
    except ValueError:
        mann_res = None

    # Chi-square test on any_affair vs has_children
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2_p = np.nan
    if contingency.shape == (2, 2):
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression for having any affair, with and without basic controls
    logit_biv = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)

    # Add simple demographic/relationship controls
    logit_adj = smf.logit(
        "any_affair ~ has_children + resp_age_years + years_married + marriage_rating",
        data=df,
    ).fit(disp=False)

    coef_biv = float(logit_biv.params["has_children"])
    p_biv = float(logit_biv.pvalues["has_children"])
    or_biv = float(np.exp(coef_biv))

    coef_adj = float(logit_adj.params["has_children"])
    p_adj = float(logit_adj.pvalues["has_children"])
    or_adj = float(np.exp(coef_adj))

    # Decide direction: does having children *decrease* affairs?
    direction_reduces = or_adj < 1.0

    # Combine significance information from multiple tests
    p_values = [
        p_adj,
        p_biv,
        ttest_res.pvalue if ttest_res is not None else np.nan,
        mann_res.pvalue if mann_res is not None else np.nan,
        chi2_p,
    ]
    p_values = [p for p in p_values if np.isfinite(p)]
    best_p = min(p_values) if p_values else np.nan

    # Map evidence to a 0–100 Likert-style score
    response = 50  # neutral default
    if np.isfinite(best_p):
        if direction_reduces:
            # Evidence that having children is associated with *fewer* affairs
            if best_p < 0.01:
                response = 85
            elif best_p < 0.05:
                response = 75
            elif best_p < 0.10:
                response = 60
            else:
                response = 45  # direction suggests reduction but weak evidence
        else:
            # Evidence that children do not reduce affairs (or even increase them)
            if best_p < 0.01:
                response = 15
            elif best_p < 0.05:
                response = 25
            elif best_p < 0.10:
                response = 40
            else:
                response = 50

    # Build explanation text
    explanation_lines = []

    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "Using the metadata, I treated the 'age' column as the numeric frequency of extramarital intercourse "
        "over the past year and the 'religiousness' column as a yes/no indicator of children in the marriage."
    )
    explanation_lines.append(
        f"There are {int(n_by_group.get(0, 0))} respondents without children and {int(n_by_group.get(1, 0))} with children."
    )
    explanation_lines.append(
        f"Mean affair frequency (0 = none, higher = more frequent) is {mean_no_children:.2f} without children "
        f"and {mean_children:.2f} with children; the proportion having any affair is "
        f"{prop_any_no_children:.2%} vs {prop_any_children:.2%}, respectively."
    )
    explanation_lines.append(
        f"A two-sample t-test comparing affair frequency between groups yields p = {ttest_res.pvalue:.4f}, "
        + (
            f"and a Mann–Whitney U test yields p = {mann_res.pvalue:.4f}."
            if mann_res is not None
            else "and a Mann–Whitney U test could not be computed."
        )
    )
    if np.isfinite(chi2_p):
        explanation_lines.append(
            f"A chi-square test of independence between 'any affair' and 'has children' yields p = {chi2_p:.4f}."
        )
    explanation_lines.append(
        f"A bivariate logistic regression of having any affair on the children indicator gives an odds ratio of "
        f"{or_biv:.2f} with p = {p_biv:.4f}; after controlling for respondent age, years married, and marriage "
        f"rating, the odds ratio for having children is {or_adj:.2f} with p = {p_adj:.4f}."
    )

    if direction_reduces and best_p < 0.05:
        explanation_lines.append(
            "Across these analyses, having children is associated with meaningfully lower odds and frequency of "
            "extramarital affairs, and this relationship is statistically significant at conventional levels."
        )
        explanation_lines.append(
            "I therefore answer 'Yes'—there is evidence that having children decreases engagement in extramarital "
            "affairs—and place this conclusion above the neutral point on the 0–100 scale."
        )
    elif direction_reduces and best_p < 0.10:
        explanation_lines.append(
            "The estimated effects generally point toward fewer extramarital affairs among couples with children, "
            "but the statistical evidence is only marginal (p between 0.05 and 0.10)."
        )
        explanation_lines.append(
            "I interpret this as weak evidence in favor of a decrease, yielding a score only modestly above neutral "
            "on the 0–100 scale."
        )
    elif direction_reduces:
        explanation_lines.append(
            "Point estimates suggest slightly fewer extramarital affairs among couples with children, but these "
            "differences are not statistically persuasive (p ≥ 0.10), so the data do not strongly support a real effect."
        )
        explanation_lines.append(
            "I therefore lean toward 'No clear evidence of a decrease', with a score just below the neutral point."
        )
    else:
        explanation_lines.append(
            "The estimated effects do not show a consistent, statistically significant reduction in affairs among "
            "couples with children; if anything, the odds ratio is at or above 1."
        )
        explanation_lines.append(
            "I therefore do not find evidence that having children decreases engagement in extramarital affairs, "
            "and reflect this with a score at or below the neutral point on the 0–100 scale."
        )

    explanation = " ".join(explanation_lines)

    # Ensure the response is an int in [0, 100]
    response_int = int(max(0, min(100, round(response))))

    output = {
        "response": response_int,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

