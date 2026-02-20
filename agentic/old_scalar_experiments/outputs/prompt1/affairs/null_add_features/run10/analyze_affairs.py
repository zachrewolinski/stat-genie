import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Ensure expected columns are present
    required_cols = [
        "affairs",
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing key variables
    df = df.dropna(subset=required_cols)

    # Binary indicator for any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children
    grouped = df.groupby("children")
    mean_affairs = grouped["affairs"].mean()
    prop_any_affair = grouped["any_affair"].mean()
    n_by_group = grouped.size()

    # Difference in means (affairs count)
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    # Welch t-test for unequal variances
    t_stat, p_ttest = stats.ttest_ind(
        affairs_yes,
        affairs_no,
        equal_var=False,
        nan_policy="omit",
    )

    # Logistic regression for probability of any affair,
    # adjusting for core demographic/marital covariates.
    # children is treated as a categorical variable.
    logit_model = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    # Extract coefficient and p-value for having children vs no children.
    # With categories 'no' and 'yes', statsmodels uses 'no' as baseline,
    # so the term is typically C(children)[T.yes].
    coef_children = None
    pval_children = None
    for term, coef in logit_model.params.items():
        if term.startswith("C(children)[T."):
            coef_children = float(coef)
            pval_children = float(logit_model.pvalues[term])
            break

    if coef_children is None or pval_children is None:
        raise RuntimeError("Could not find children coefficient in logistic model.")

    # Decide on answer:
    # - We say "Yes" only if the adjusted association suggests that
    #   having children is associated with *lower* odds of any affair
    #   (negative coefficient vs baseline no-children) and this effect
    #   is statistically meaningful (p < 0.05).
    # - Otherwise, we answer "No" to reflect lack of clear evidence
    #   that children decrease engagement in extramarital affairs.
    alpha = 0.05
    if coef_children < 0 and pval_children < alpha:
        response = "Yes"
    else:
        response = "No"

    # Build explanation text
    mean_affairs_yes = float(mean_affairs.get("yes", np.nan))
    mean_affairs_no = float(mean_affairs.get("no", np.nan))
    prop_any_yes = float(prop_any_affair.get("yes", np.nan))
    prop_any_no = float(prop_any_affair.get("no", np.nan))

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs? "
        "Using the provided survey data (601 married individuals), I compared the frequency and "
        "probability of extramarital affairs between marriages with and without children. "
        f"In the raw data, the average number of affairs in the past year was "
        f"{mean_affairs_yes:.2f} for couples with children (n={int(n_by_group.get('yes', 0))}) "
        f"and {mean_affairs_no:.2f} for couples without children (n={int(n_by_group.get('no', 0))}). "
        f"The proportion who had any affair at all was {prop_any_yes:.2%} with children versus "
        f"{prop_any_no:.2%} without children. A Welch t-test comparing the mean affair counts "
        f"between the two groups yielded p={p_ttest:.3f}. "
        "To adjust for other factors that may influence infidelity, I fit a logistic regression "
        "model for the probability of having any affair, including covariates for age, years "
        "married, religiousness, education, occupation, marital satisfaction rating, and gender. "
        f"In this model, the coefficient for having children (relative to no children) was "
        f"{coef_children:.3f} with p-value {pval_children:.3f}. "
        "A negative, statistically significant coefficient would indicate that having children is "
        "associated with meaningfully lower odds of engaging in an extramarital affair, after "
        "controlling for these other variables. "
    )

    if response == "Yes":
        explanation += (
            "Because the estimated children effect is negative and statistically significant, "
            "the data provide evidence that having children is associated with lower engagement "
            "in extramarital affairs in this sample."
        )
    else:
        explanation += (
            "In this analysis, the children coefficient is not both negative and statistically "
            "significant at the 5% level, so the data do not provide clear evidence that having "
            "children decreases engagement in extramarital affairs. Accordingly, I conclude that "
            "we cannot say that having children reduces extramarital affairs in this dataset."
        )

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

