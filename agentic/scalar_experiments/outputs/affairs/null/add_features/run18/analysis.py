import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Basic cleaning / derived variables
    # Outcome as count and as "any affair" indicator
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure children and gender are treated as categories
    df["children"] = df["children"].astype("category")
    df["gender"] = df["gender"].astype("category")

    print("=== Sample size ===")
    print(len(df))

    print("\n=== Children value counts ===")
    print(df["children"].value_counts(dropna=False))

    print("\n=== Mean affairs by children ===")
    print(df.groupby("children")["affairs"].mean())

    print("\n=== Any-affair proportion by children ===")
    print(df.groupby("children")["any_affair"].mean())

    # Logistic regression for any affair, with common covariates
    # Use a modest set of plausible controls that exist in this file.
    formula = "any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"

    print("\n=== Logistic regression: any_affair on children + controls ===")
    try:
        logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
        print(logit_model.summary())

        # Extract children effect
        params = logit_model.params
        conf_int = logit_model.conf_int()
        pvalues = logit_model.pvalues

        # children is coded as C(children)[T.yes] with baseline "no"
        child_term = "C(children)[T.yes]"
        if child_term in params.index:
            coef = params[child_term]
            ci_low, ci_high = conf_int.loc[child_term]
            pval = pvalues[child_term]
            odds_ratio = np.exp(coef)
            or_low, or_high = np.exp(ci_low), np.exp(ci_high)

            print("\nEffect of having children (yes vs no):")
            print(f"  Log-odds coef: {coef:.4f}")
            print(f"  95% CI (log-odds): [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"  Odds ratio: {odds_ratio:.3f}")
            print(f"  95% CI (OR): [{or_low:.3f}, {or_high:.3f}]")
            print(f"  p-value: {pval:.4g}")
        else:
            print("\nChildren term not found in model parameters.")
    except Exception as exc:  # pragma: no cover - defensive
        print("Logistic regression failed:", repr(exc))

    # Linear regression on the raw affairs count as a secondary check
    print("\n=== Linear regression: affairs on children + controls ===")
    try:
        ols_model = smf.ols(
            formula="affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
            data=df,
        ).fit()
        print(ols_model.summary())
    except Exception as exc:  # pragma: no cover - defensive
        print("Linear regression failed:", repr(exc))


if __name__ == "__main__":
    main()

