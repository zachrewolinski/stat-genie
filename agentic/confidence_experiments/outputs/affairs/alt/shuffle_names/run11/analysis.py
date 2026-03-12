import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json, the variable describing extramarital affairs
    # frequency is stored in the column currently named "age".
    affairs_freq = df["age"]

    # The description for the column currently named "religiousness"
    # indicates it is actually "Are there children in the marriage?".
    # Encode as 1 = has children, 0 = no children.
    children = df["religiousness"].map({"yes": 1, "no": 0}).astype(int)

    # Create a binary indicator for having any extramarital affair
    any_affair = (affairs_freq > 0).astype(int)

    # Reconstruct key covariates based on the descriptions in info.json.
    # - "occupation" column is described as age in years.
    # - "children" column is described as years married.
    # - "rating" column is described as religiousness (1–5).
    # - "affairs" column is described as marriage rating (1–5).
    age_years = df["occupation"]
    years_married = df["children"]
    religiosity = df["rating"]
    marital_rating = df["affairs"]

    # Descriptive comparison: proportion with any affair by children status.
    crosstab = (
        pd.crosstab(children, any_affair, normalize="index")
        .rename(index={0: "no_children", 1: "children"})
        .rename(columns={0: "no_affair", 1: "any_affair"})
    )

    # Unadjusted logistic regression: any affair ~ children.
    X_simple = sm.add_constant(pd.DataFrame({"children": children}))
    model_simple = sm.Logit(any_affair, X_simple).fit(disp=False)
    coef_children_simple = model_simple.params["children"]
    pval_children_simple = model_simple.pvalues["children"]
    or_children_simple = float(np.exp(coef_children_simple))

    # Adjusted logistic regression with key covariates to account for
    # potential confounding.
    X_full = pd.DataFrame(
        {
            "children": children,
            "age_years": age_years,
            "years_married": years_married,
            "religiosity": religiosity,
            "marital_rating": marital_rating,
        }
    )
    X_full = sm.add_constant(X_full)
    model_full = sm.Logit(any_affair, X_full).fit(disp=False)
    coef_children_full = model_full.params["children"]
    pval_children_full = model_full.pvalues["children"]
    or_children_full = float(np.exp(coef_children_full))

    # Decide on the Likert-scale response (0–100).
    # Interpretation:
    # - Direction: negative coefficient => children associated with fewer affairs.
    # - Strength: driven by odds ratio and p-values from both models.
    decreases_affairs = coef_children_full < 0

    # Base around an agnostic midpoint, then shift based on evidence.
    response_score = 50

    if decreases_affairs:
        # Effect suggests a protective association.
        if pval_children_full < 0.01 and or_children_full < 0.7:
            response_score = 85
        elif pval_children_full < 0.05 and or_children_full < 0.85:
            response_score = 75
        elif pval_children_full < 0.1 and or_children_full < 0.9:
            response_score = 65
        else:
            # Weak or non-significant evidence in the expected direction.
            response_score = 55
    else:
        # Either no effect, or children associated with more affairs.
        if pval_children_full < 0.05 and or_children_full > 1.15:
            response_score = 20
        elif pval_children_full < 0.1 and or_children_full > 1.05:
            response_score = 35
        else:
            # Essentially no clear evidence for a decrease.
            response_score = 45

    # Construct a human-readable explanation summarizing key evidence.
    no_children_affair_rate = crosstab.loc["no_children", "any_affair"]
    children_affair_rate = crosstab.loc["children", "any_affair"]

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs? "
        "Using the provided survey data (n = {n}), I defined 'having any affair' as reporting a "
        "non-zero frequency of extramarital intercourse over the past year (from the column "
        "described as affair frequency) and treated the column described as 'Are there children in "
        "the marriage?' as a binary children indicator. "
        "Descriptively, the share of respondents reporting any affair was "
        "{no_children_rate:.1%} among those without children and "
        "{children_rate:.1%} among those with children. "
        "I then fit a logistic regression with any affair as the outcome and children as the main "
        "predictor. In the unadjusted model, the odds ratio for having children was "
        "{or_simple:.2f} (p = {p_simple:.3f}). "
        "To account for potential confounding, I fit a second logistic model adjusting for age in "
        "years, years married, religiousness, and self-rated marital quality (reconstructed from "
        "the variable descriptions in the metadata). In this adjusted model, the odds ratio for "
        "having children was {or_full:.2f} (p = {p_full:.3f}). "
        "These results indicate that "
    ).format(
        n=len(df),
        no_children_rate=no_children_affair_rate,
        children_rate=children_affair_rate,
        or_simple=or_children_simple,
        p_simple=pval_children_simple,
        or_full=or_children_full,
        p_full=pval_children_full,
    )

    if decreases_affairs and pval_children_full < 0.05:
        explanation += (
            "having children is statistically significantly associated with a lower likelihood of "
            "engaging in extramarital affairs, even after adjusting for these covariates, although "
            "the magnitude of the reduction is captured by the reported odds ratio."
        )
    elif decreases_affairs:
        explanation += (
            "the estimated effect of having children points toward a lower likelihood of affairs, "
            "but this association is not conventionally statistically significant, so the evidence "
            "for a protective effect is weak."
        )
    elif pval_children_full < 0.05:
        explanation += (
            "having children is statistically significantly associated with an increased likelihood "
            "of extramarital affairs, contrary to the hypothesized decrease."
        )
    else:
        explanation += (
            "there is no statistically convincing evidence that having children either increases or "
            "decreases the likelihood of extramarital affairs; estimated differences are small and "
            "not reliably different from zero."
        )

    conclusion = {"response": int(response_score), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

