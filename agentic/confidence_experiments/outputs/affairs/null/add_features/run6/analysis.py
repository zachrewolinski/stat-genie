import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Basic cleaning: keep rows with non-missing key variables
    df = df.dropna(subset=["affairs", "children"])

    # Create a binary any-affair indicator: 0 = none, 1 = any affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = yes, 0 = no
    df["children_bin"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Quick descriptive statistics
    mean_affairs_children = df.loc[df["children_bin"] == 1, "affairs"].mean()
    mean_affairs_no_children = df.loc[df["children_bin"] == 0, "affairs"].mean()

    prop_any_children = df.loc[df["children_bin"] == 1, "any_affair"].mean()
    prop_any_no_children = df.loc[df["children_bin"] == 0, "any_affair"].mean()

    # Logistic regression of any_affair on children indicator and some controls
    # These controls are standard in analyses of this dataset.
    formula = "any_affair ~ children_bin + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        children_coef = model.params.get("children_bin", float("nan"))
        children_pval = model.pvalues.get("children_bin", float("nan"))
        odds_ratio = float("nan")
        if pd.notnull(children_coef):
            odds_ratio = float(np.exp(children_coef))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Logistic regression failed: {exc}")
        model = None
        children_coef = float("nan")
        children_pval = float("nan")
        odds_ratio = float("nan")

    print("Descriptive statistics:")
    print(f"  Mean affairs (children = yes): {mean_affairs_children:.3f}")
    print(f"  Mean affairs (children = no):  {mean_affairs_no_children:.3f}")
    print(f"  Proportion with any affair (children = yes): {prop_any_children:.3f}")
    print(f"  Proportion with any affair (children = no):  {prop_any_no_children:.3f}")

    if model is not None:
        print("\nLogistic regression of any_affair on children and controls:")
        print(model.summary())
        print(f"\nchildren_bin coefficient: {children_coef:.4f}")
        print(f"children_bin odds ratio: {odds_ratio:.4f}")
        print(f"children_bin p-value: {children_pval:.4g}")
    else:
        print("\nLogistic regression model not available.")

    # Poisson regression on the affair counts as an additional check
    poisson_formula = "affairs ~ children_bin + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"
    try:
        poisson_model = smf.glm(
            formula=poisson_formula,
            data=df,
            family=sm.families.Poisson(),
        ).fit(cov_type="HC0")
        pois_children_coef = poisson_model.params.get("children_bin", float("nan"))
        pois_children_pval = poisson_model.pvalues.get("children_bin", float("nan"))
        pois_rate_ratio = float("nan")
        if pd.notnull(pois_children_coef):
            pois_rate_ratio = float(np.exp(pois_children_coef))

        print("\nPoisson regression of affair counts on children and controls:")
        print(poisson_model.summary())
        print(f"\nPoisson children_bin coefficient: {pois_children_coef:.4f}")
        print(f"Poisson children_bin rate ratio: {pois_rate_ratio:.4f}")
        print(f"Poisson children_bin p-value: {pois_children_pval:.4g}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\nPoisson regression failed: {exc}")


if __name__ == "__main__":
    main()
