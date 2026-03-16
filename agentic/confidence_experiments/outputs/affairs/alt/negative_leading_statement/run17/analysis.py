import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Create binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children
    group = df.groupby("children", observed=True)
    summary = group["affairs"].agg(["count", "mean"])
    prop_affair = group["has_affair"].mean()

    print("=== Descriptive statistics by children ===")
    print(summary)
    print("\nProportion with any affair by children:")
    print(prop_affair)

    # Simple logistic regression: any affair ~ children
    print("\n=== Logistic regression: has_affair ~ C(children) ===")
    model_simple = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print(model_simple.summary())
    coef_child_simple = model_simple.params.get("C(children)[T.yes]", np.nan)
    p_child_simple = model_simple.pvalues.get("C(children)[T.yes]", np.nan)
    or_child_simple = float(np.exp(coef_child_simple)) if np.isfinite(coef_child_simple) else np.nan
    print(
        f"\nSimple model children[T.yes]: coef={coef_child_simple:.4f}, "
        f"OR={or_child_simple:.4f}, p={p_child_simple:.4g}"
    )

    # Adjusted logistic regression with key covariates
    print("\n=== Logistic regression: has_affair ~ C(children) + covariates ===")
    formula_full = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    print(model_full.summary())
    coef_child_full = model_full.params.get("C(children)[T.yes]", np.nan)
    p_child_full = model_full.pvalues.get("C(children)[T.yes]", np.nan)
    or_child_full = float(np.exp(coef_child_full)) if np.isfinite(coef_child_full) else np.nan
    print(
        f"\nAdjusted model children[T.yes]: coef={coef_child_full:.4f}, "
        f"OR={or_child_full:.4f}, p={p_child_full:.4g}"
    )

    # Also report difference in observed proportions
    prop_yes = float(prop_affair.get("yes", np.nan))
    prop_no = float(prop_affair.get("no", np.nan))
    diff_prop = prop_yes - prop_no
    print(
        f"\nObserved probability of any affair: "
        f"children=no: {prop_no:.3f}, children=yes: {prop_yes:.3f}, "
        f"difference (yes - no) = {diff_prop:.3f}"
    )


if __name__ == "__main__":
    main()

