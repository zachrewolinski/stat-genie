import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic sanity checks
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Create binary outcome: any affair vs none
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children
    group = df.groupby("children", observed=True)

    desc = group["affairs"].agg(["mean", "median", "std", "count"])
    prop_any = group["has_affair"].mean()

    print("\nAffairs count by children status:")
    print(desc)
    print("\nProportion with any affair by children status:")
    print(prop_any)

    # Logistic regression: has_affair ~ children + controls
    formula_logit = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula_logit, data=df).fit(disp=False)
    print("\nLogistic regression results (has_affair ~ children + controls):")
    print(logit_model.summary())

    # Extract coefficient and p-value for children effect
    params = logit_model.params
    pvalues = logit_model.pvalues
    children_terms = {k: (params[k], pvalues[k]) for k in params.index if "children" in k}
    print("\nChildren-related coefficients in logistic model:")
    for name, (coef, pval) in children_terms.items():
        odds_ratio = float(np.exp(coef))
        print(
            f"{name}: coef={coef:.4f}, odds_ratio={odds_ratio:.3f}, "
            f"p-value={pval:.4g}"
        )

    # Poisson regression on affair counts (for robustness)
    formula_pois = (
        "affairs ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    pois_model = smf.glm(
        formula=formula_pois,
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    print("\nPoisson regression results (affairs count ~ children + controls):")
    print(pois_model.summary())

    params_p = pois_model.params
    pvalues_p = pois_model.pvalues
    children_terms_p = {
        k: (params_p[k], pvalues_p[k]) for k in params_p.index if "children" in k
    }
    print("\nChildren-related coefficients in Poisson model:")
    for name, (coef, pval) in children_terms_p.items():
        irr = float(np.exp(coef))
        print(
            f"{name}: coef={coef:.4f}, IRR={irr:.3f}, "
            f"p-value={pval:.4g}"
        )


if __name__ == "__main__":
    main()
