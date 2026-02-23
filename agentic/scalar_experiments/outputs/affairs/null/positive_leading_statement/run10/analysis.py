import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic structure
    print("Rows, columns:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # Descriptive stats by children
    print("\nValue counts for children:")
    print(df["children"].value_counts())

    print("\nMean number of affairs by children:")
    print(df.groupby("children")["affairs"].agg(["mean", "std", "median", "count"]))

    # Binary indicator: any affair vs none
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    print("\nProportion with any affair by children:")
    print(df.groupby("children")["any_affair"].mean())

    # Logistic regression for having any affair
    formula_logit = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + C(gender) + occupation + rating"
    )
    logit_model = smf.logit(formula=formula_logit, data=df).fit(disp=False)

    print("\nLogistic regression: any_affair on children and covariates")
    print(logit_model.summary())

    # Extract effect of having children
    coef_children = logit_model.params.get("C(children)[T.yes]", np.nan)
    pval_children = logit_model.pvalues.get("C(children)[T.yes]", np.nan)
    print("\nChildren effect (logit):")
    print("coef:", coef_children, "p-value:", pval_children)

    # Marginal effect on predicted probability at mean covariates
    mean_row = df[
        ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    ].mean()
    base_data = pd.DataFrame(
        {
            "children": ["no", "yes"],
            "gender": ["male", "male"],
            "age": [mean_row["age"]] * 2,
            "yearsmarried": [mean_row["yearsmarried"]] * 2,
            "religiousness": [mean_row["religiousness"]] * 2,
            "education": [mean_row["education"]] * 2,
            "occupation": [mean_row["occupation"]] * 2,
            "rating": [mean_row["rating"]] * 2,
        }
    )
    preds = logit_model.predict(base_data)
    print("\nPredicted probability of any affair at mean covariates:")
    for row, prob in zip(base_data.itertuples(index=False), preds):
        print(f"children={row.children}, predicted_prob_any_affair={prob:.3f}")

    # Poisson regression on count of affairs as a robustness check
    formula_pois = (
        "affairs ~ C(children) + age + yearsmarried + religiousness "
        "+ education + C(gender) + occupation + rating"
    )
    pois_model = smf.glm(
        formula=formula_pois, data=df, family=sm.families.Poisson()
    ).fit()

    print("\nPoisson regression: affairs count on children and covariates")
    print(pois_model.summary())

    coef_children_pois = pois_model.params.get("C(children)[T.yes]", np.nan)
    pval_children_pois = pois_model.pvalues.get("C(children)[T.yes]", np.nan)
    print("\nChildren effect (Poisson):")
    print("coef:", coef_children_pois, "p-value:", pval_children_pois)

    # Also compute observed rate ratio of mean affairs between groups
    means = df.groupby("children")["affairs"].mean()
    if "yes" in means and "no" in means and means["no"] > 0:
        rate_ratio = means["yes"] / means["no"]
        print("\nObserved mean affairs by children and rate ratio (yes/no):")
        print(means)
        print("rate_ratio_yes_over_no:", rate_ratio)


if __name__ == "__main__":
    main()

