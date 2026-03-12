import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramer's V effect size for a chi-square test."""
    if n <= 0:
        return float("nan")
    k = min(r - 1, c - 1)
    if k <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * k)))


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df_social = df[df["y"] != 1].copy()
    df_social["majority"] = (df_social["y"] == 2).astype(int)

    n = len(df)

    results = {"n": int(n), "descriptives": {}, "tests": {}}

    # Descriptives for outcome by culture and age (age treated as categorical for descriptives)
    results["descriptives"]["social_by_culture"] = (
        pd.crosstab(df["culture"], df["social"], normalize="index")
        .rename_axis("culture")
        .reset_index()
        .to_dict(orient="records")
    )
    results["descriptives"]["social_by_age"] = (
        pd.crosstab(df["age"], df["social"], normalize="index")
        .rename_axis("age")
        .reset_index()
        .to_dict(orient="records")
    )

    results["descriptives"]["majority_by_culture"] = (
        pd.crosstab(df_social["culture"], df_social["majority"], normalize="index")
        .rename_axis("culture")
        .reset_index()
        .to_dict(orient="records")
    )
    results["descriptives"]["majority_by_age"] = (
        pd.crosstab(df_social["age"], df_social["majority"], normalize="index")
        .rename_axis("age")
        .reset_index()
        .to_dict(orient="records")
    )

    # Chi-square tests of association
    tests = {}

    # Social vs culture
    ct = pd.crosstab(df["culture"], df["social"])
    chi2, p, dof, _ = chi2_contingency(ct)
    tests["social_culture_chi2"] = {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v(chi2, n, *ct.shape)),
    }

    # Social vs age (age treated as categorical)
    ct = pd.crosstab(df["age"], df["social"])
    chi2, p, dof, _ = chi2_contingency(ct)
    tests["social_age_chi2"] = {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v(chi2, n, *ct.shape)),
    }

    # Majority vs culture
    ct = pd.crosstab(df_social["culture"], df_social["majority"])
    chi2, p, dof, _ = chi2_contingency(ct)
    tests["majority_culture_chi2"] = {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v(chi2, len(df_social), *ct.shape)),
    }

    # Majority vs age
    ct = pd.crosstab(df_social["age"], df_social["majority"])
    chi2, p, dof, _ = chi2_contingency(ct)
    tests["majority_age_chi2"] = {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v(chi2, len(df_social), *ct.shape)),
    }

    # Logistic regressions
    tests["logit"] = {}

    # Logistic: reliance on social information
    model_social = smf.logit("social ~ C(culture) + age", data=df).fit(disp=False)
    tests["logit"]["social"] = {
        "age_coef": float(model_social.params["age"]),
        "age_p": float(model_social.pvalues["age"]),
        "culture_pvalues": {
            name: float(pval)
            for name, pval in model_social.pvalues.items()
            if name.startswith("C(culture)")
        },
        "pseudo_r2": float(model_social.prsquared),
    }

    # Logistic: majority vs minority, conditional on social learning
    model_majority = smf.logit("majority ~ C(culture) + age", data=df_social).fit(
        disp=False
    )
    tests["logit"]["majority"] = {
        "age_coef": float(model_majority.params["age"]),
        "age_p": float(model_majority.pvalues["age"]),
        "culture_pvalues": {
            name: float(pval)
            for name, pval in model_majority.pvalues.items()
            if name.startswith("C(culture)")
        },
        "pseudo_r2": float(model_majority.prsquared),
    }

    results["tests"] = tests

    # Save raw analysis results for inspection
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

