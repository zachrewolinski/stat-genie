import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create binary indicator of any extramarital affair in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    group = df.groupby("children")
    desc = group[["affairs", "affair_any"]].agg(["mean", "std", "sum", "count"])
    # Flatten MultiIndex columns for JSON serialization
    desc_flat = desc.copy()
    desc_flat.columns = [
        "_".join(str(c) for c in col).strip() for col in desc_flat.columns.to_flat_index()
    ]
    desc_dict = desc_flat.to_dict(orient="index")

    # Chi-square test of independence between children and any affair
    contingency = pd.crosstab(df["children"], df["affair_any"])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    # Unadjusted logistic regression: any affair ~ children
    logit_simple = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)

    # Adjusted logistic regression with standard covariates
    logit_full_formula = (
        "affair_any ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_full = smf.logit(logit_full_formula, data=df).fit(disp=False)

    # Poisson regression on counts as a complementary check
    poisson_full = smf.glm(
        "affairs ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating",
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    # Extract key effect estimates for children=yes vs no
    coef_simple = logit_simple.params.get("C(children)[T.yes]", np.nan)
    p_simple = logit_simple.pvalues.get("C(children)[T.yes]", np.nan)
    or_simple = float(np.exp(coef_simple)) if np.isfinite(coef_simple) else np.nan

    coef_full = logit_full.params.get("C(children)[T.yes]", np.nan)
    p_full = logit_full.pvalues.get("C(children)[T.yes]", np.nan)
    or_full = float(np.exp(coef_full)) if np.isfinite(coef_full) else np.nan

    coef_pois = poisson_full.params.get("C(children)[T.yes]", np.nan)
    p_pois = poisson_full.pvalues.get("C(children)[T.yes]", np.nan)
    rr_pois = float(np.exp(coef_pois)) if np.isfinite(coef_pois) else np.nan

    results = {
        "descriptives_by_children": desc_dict,
        "chi2_children_affair_any": {
            "chi2": float(chi2),
            "p_value": float(chi2_p),
            "dof": int(dof),
        },
        "logit_simple_children": {
            "coef": float(coef_simple),
            "p_value": float(p_simple),
            "odds_ratio": float(or_simple),
        },
        "logit_full_children": {
            "coef": float(coef_full),
            "p_value": float(p_full),
            "odds_ratio": float(or_full),
        },
        "poisson_full_children": {
            "coef": float(coef_pois),
            "p_value": float(p_pois),
            "rate_ratio": float(rr_pois),
        },
    }

    # Print JSON to stdout so it can be inspected from the outside.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
