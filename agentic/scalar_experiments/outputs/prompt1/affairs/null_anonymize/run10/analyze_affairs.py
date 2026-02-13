import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in the marriage (1 = yes, 0 = no)
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Basic descriptive statistics by children status
    affair_by_children = df.groupby("children")["any_affair"].agg(["mean", "sum", "count"])
    freq_by_children = df.groupby("children")["feature2"].mean()

    rate_children = float(affair_by_children.loc[1, "mean"])
    rate_no_children = float(affair_by_children.loc[0, "mean"])
    mean_freq_children = float(freq_by_children.loc[1])
    mean_freq_no_children = float(freq_by_children.loc[0])

    # Default values in case the regression fails
    coeff_children = None
    pvalue_children = None
    prob_children = None
    prob_no_children = None
    diff_prob = None
    logistic_ok = False

    # Logistic regression controlling for key covariates
    try:
        X = df[
            [
                "children",
                "feature4",  # age
                "feature5",  # years married
                "feature7",  # religiousness
                "feature8",  # education
                "feature9",  # occupation
                "feature10",  # marriage rating
            ]
        ].copy()
        X = sm.add_constant(X)
        y = df["any_affair"]

        model = sm.Logit(y, X).fit(disp=False)

        coeff_children = float(model.params["children"])
        pvalue_children = float(model.pvalues["children"])

        # Predicted probabilities at the mean covariate values,
        # toggling only the children indicator.
        mean_vals = X.mean()
        mean_no_children = mean_vals.copy()
        mean_no_children["children"] = 0
        mean_children = mean_vals.copy()
        mean_children["children"] = 1

        prob_no_children = float(model.predict(mean_no_children))
        prob_children = float(model.predict(mean_children))
        diff_prob = prob_children - prob_no_children

        logistic_ok = True
    except Exception:
        # Fall back on descriptive stats only
        logistic_ok = False

    # Decision logic: strong evidence children decrease affairs only if
    # (a) descriptive rates are lower with children, and
    # (b) adjusted logistic effect is negative and significant.
    decreases = False
    if logistic_ok:
        if (
            rate_children < rate_no_children
            and diff_prob is not None
            and diff_prob < 0
            and coeff_children is not None
            and coeff_children < 0
            and pvalue_children is not None
            and pvalue_children < 0.05
        ):
            decreases = True
    else:
        # If regression fails, use descriptive rates only.
        if rate_children < rate_no_children and mean_freq_children < mean_freq_no_children:
            decreases = True

    response = "Yes" if decreases else "No"

    # Build explanation string with the key numerical evidence.
    desc_part = (
        "Using the Fair (1978) affairs survey (n=601), "
        "I defined engagement in extramarital affairs as any non-zero value of feature2 "
        "(frequency of extramarital intercourse in the past year) and compared respondents "
        "with and without children (feature6, 'yes'/'no'). "
        f"Empirically, {rate_children:.3f} of respondents with children versus "
        f"{rate_no_children:.3f} without children reported any affair; "
        f"mean affair frequency was {mean_freq_children:.3f} with children "
        f"and {mean_freq_no_children:.3f} without children. "
    )

    if logistic_ok:
        direction = "lower" if coeff_children is not None and coeff_children < 0 else "higher"
        logit_part = (
            "I then fit a logistic regression of any affair on the children indicator, "
            "controlling for age (feature4), years married (feature5), religiousness (feature7), "
            "education (feature8), occupation (feature9), and self-rated marital happiness "
            "(feature10). "
        )
        if coeff_children is not None and pvalue_children is not None and diff_prob is not None:
            logit_part += (
                f"The estimated coefficient on having children was {coeff_children:.3f} "
                f"(p={pvalue_children:.3f}), implying a predicted affair probability of "
                f"{prob_children:.3f} with children versus {prob_no_children:.3f} without, "
                f"a {diff_prob:.3f} {direction} probability when holding other factors constant. "
            )
    else:
        logit_part = (
            "A multivariable logistic regression adjustment was attempted but did not converge; "
            "the conclusion is therefore based on the descriptive differences alone. "
        )

    if response == "Yes":
        conclusion_part = (
            "Because having children is associated with both a lower raw incidence of any affairs "
            "and a negative, statistically significant effect on affair probability after adjusting "
            "for major demographic and marital factors, I conclude that the data support the claim "
            "that having children decreases engagement in extramarital affairs."
        )
    else:
        conclusion_part = (
            "Given that the presence of children is not consistently associated with a lower "
            "incidence or probability of affairs, and the adjusted effect is not both negative "
            "and statistically significant at conventional levels, there is insufficient evidence "
            "to claim that having children decreases engagement in extramarital affairs in this sample."
        )

    explanation = desc_part + logit_part + conclusion_part

    result = {
        "response": response,
        "explanation": explanation,
    }

    # Write result as a single JSON object with no extra lines.
    with open("conclusion.txt", "w") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    main()

