import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # Create binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children indicator
    df["children_yes"] = (df["children"] == "yes").astype(int)

    n = len(df)
    counts_children = df["children"].value_counts().to_dict()

    # Simple descriptive statistics
    mean_affairs_by_children = df.groupby("children")["affairs"].mean().to_dict()
    prop_has_affair_by_children = df.groupby("children")["has_affair"].mean().to_dict()

    # Logistic regression for having any affair, controlling for covariates
    # children is entered as a categorical predictor so we can directly
    # interpret the effect of "children = yes" relative to "no".
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + rating + C(gender) + C(occupation)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    params = logit_model.params
    pvalues = logit_model.pvalues

    # Effect of having children (yes vs no)
    child_term = "C(children)[T.yes]"
    child_coef = float(params.get(child_term, np.nan))
    child_p = float(pvalues.get(child_term, np.nan))
    child_or = float(np.exp(child_coef)) if np.isfinite(child_coef) else np.nan

    print("N observations:", n)
    print("Counts by children:", counts_children)
    print("Mean affairs by children:", mean_affairs_by_children)
    print("Proportion with any affair by children:", prop_has_affair_by_children)
    print()
    print("Logistic regression: has_affair ~ C(children) + controls")
    print("Children term:", child_term)
    print(f"  Coefficient: {child_coef:.4f}")
    print(f"  Odds ratio: {child_or:.4f}")
    print(f"  p-value: {child_p:.4g}")

    # Also print full summary for detailed inspection if needed
    print("\nFull model summary:")
    print(logit_model.summary())


if __name__ == "__main__":
    main()

