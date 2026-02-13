import json
from math import exp

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Map columns to their semantic meaning based on info.json.
    # age: frequency of extramarital intercourse in past year (0 = none, >0 = some)
    # religiousness: yes/no indicator for whether there are children in the marriage.
    df["any_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = (df["religiousness"] == "yes").astype(int)

    # Descriptive statistics: affair rates by children status.
    rates = (
        df.groupby("has_children")["any_affair"]
        .agg(["mean", "count", "sum"])
        .rename(index={0: "no_children", 1: "has_children"})
    )

    print("Affair engagement (any affair) by children status:")
    print(rates)

    # Simple logistic regression: any_affair ~ has_children
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]
    model = sm.Logit(y, X, missing="drop")
    result = model.fit(disp=False)

    coef_children = result.params["has_children"]
    p_value = result.pvalues["has_children"]
    odds_ratio = exp(coef_children)

    print("\nLogistic regression result (any_affair ~ has_children):")
    print(result.summary())
    print(f"\nCoefficient for has_children: {coef_children:.3f}")
    print(f"Odds ratio for has_children: {odds_ratio:.3f}")
    print(f"p-value for has_children: {p_value:.4g}")

    # Interpret direction: OR < 1 => children associated with fewer affairs.
    decreases_affairs = odds_ratio < 1
    statistically_significant = p_value < 0.05

    # Decide Yes/No: only answer "Yes" if there is a statistically significant
    # decrease in affair odds for people with children.
    if decreases_affairs and statistically_significant:
        response = "Yes"
        confidence = 90
    else:
        response = "No"
        if not statistically_significant and abs(odds_ratio - 1.0) < 0.1:
            # Effect is tiny and non-significant.
            confidence = 80
        elif not statistically_significant:
            confidence = 70
        else:
            # Statistically significant but in the opposite direction.
            confidence = 85

    # Build explanation text based on the descriptive and model results.
    rate_no_children = rates.loc["no_children", "mean"]
    rate_children = rates.loc["has_children", "mean"]
    n_no_children = int(rates.loc["no_children", "count"])
    n_children = int(rates.loc["has_children", "count"])

    explanation = (
        "Using the provided dataset of 601 married individuals, I treated the 'age' "
        "column as the frequency of extramarital intercourse in the past year and "
        "constructed a binary outcome indicating whether a respondent reported at "
        "least one affair. I used the 'religiousness' column as an indicator of "
        "whether there are children in the marriage (values 'yes' or 'no'). "
        f"Among respondents without children (n={n_no_children}), the proportion "
        f"reporting any affair was {rate_no_children:.3f}, while among those with "
        f"children (n={n_children}) the proportion was {rate_children:.3f}. "
        "I then fit a logistic regression model with any-affair status as the "
        "outcome and a single predictor for having children. The estimated odds "
        f"ratio for having children was {odds_ratio:.3f} with p-value {p_value:.4g}. "
        + (
            "This odds ratio is below 1, but the p-value is large, so the data do not provide statistically reliable evidence that having children reduces extramarital affairs."
            if decreases_affairs and not statistically_significant
            else "This odds ratio is above 1 and statistically significant, indicating higher odds of affairs for people with children."
            if not decreases_affairs and statistically_significant
            else "This pattern does not show a clear or statistically reliable reduction in affairs among people with children."
        )
        + " Based on the direction and statistical strength of this association, I "
        "translated the findings into a binary Yes/No answer and an accompanying "
        "confidence score on a 0–100 scale."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()
