import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def fit_logit(formula: str, data: pd.DataFrame):
    try:
        model = smf.logit(formula=formula, data=data)
        result = model.fit(disp=False)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to fit model '{formula}': {exc}")
        return None


def summarize_logit_effects(result, var_prefix: str):
    """Return min p-value and typical odds ratio magnitude for a given variable (or family of dummies)."""
    if result is None:
        return None
    pvals = result.pvalues
    params = result.params

    rel_pvals = []
    or_mags = []
    for name, p in pvals.items():
        if name.startswith(var_prefix):
            rel_pvals.append(float(p))
            or_mags.append(float(np.exp(params[name])))

    if not rel_pvals:
        return None

    min_p = float(min(rel_pvals))
    median_or = float(np.median(or_mags))
    return {"min_p": min_p, "median_or": median_or}


def chi_square_test(table: pd.DataFrame):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}


def main():
    df = pd.read_csv("boxes.csv")

    # Create derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    n_total = len(df)
    n_social = int(df["social"].sum())
    n_majority = int(df_social["majority_choice"].sum())

    print(f"Total N: {n_total}")
    print(f"Social choices (majority or minority): {n_social} ({n_social / n_total:.3f})")
    print(
        f"Majority choices among social: {n_majority} "
        f"({n_majority / len(df_social) if len(df_social) > 0 else np.nan:.3f})"
    )

    # Logistic regression: reliance on social information
    formula_social = "social ~ age + C(culture) + C(gender) + C(majority_first)"
    res_social = fit_logit(formula_social, df)

    age_p_social = float(res_social.pvalues.get("age", np.nan)) if res_social is not None else np.nan
    culture_eff_social = summarize_logit_effects(res_social, "C(culture)")

    print("\nSocial reliance model:")
    if res_social is not None:
        print(res_social.summary())
    print(f"Age effect p-value (social): {age_p_social:.4g}")
    if culture_eff_social:
        print(
            f"Culture effects (social): min p={culture_eff_social['min_p']:.4g}, "
            f"median OR={culture_eff_social['median_or']:.3f}"
        )

    # Chi-square for social reliance by culture and age (binned)
    social_by_culture = pd.crosstab(df["culture"], df["social"])
    social_culture_chi = chi_square_test(social_by_culture)
    age_bins = pd.cut(df["age"], bins=[4, 6, 8, 10, 12, 14], include_lowest=True)
    social_by_agebin = pd.crosstab(age_bins, df["social"])
    social_age_chi = chi_square_test(social_by_agebin)

    print("\nChi-square social ~ culture:", social_culture_chi)
    print("Chi-square social ~ age_bin:", social_age_chi)

    # Logistic regression: majority preference (only among social learners)
    formula_majority = "majority_choice ~ age + C(culture) + C(gender) + C(majority_first)"
    res_majority = fit_logit(formula_majority, df_social)

    age_p_majority = float(res_majority.pvalues.get("age", np.nan)) if res_majority is not None else np.nan
    culture_eff_majority = summarize_logit_effects(res_majority, "C(culture)")

    print("\nMajority preference model (conditional on social choice):")
    if res_majority is not None:
        print(res_majority.summary())
    print(f"Age effect p-value (majority): {age_p_majority:.4g}")
    if culture_eff_majority:
        print(
            f"Culture effects (majority): min p={culture_eff_majority['min_p']:.4g}, "
            f"median OR={culture_eff_majority['median_or']:.3f}"
        )

    majority_by_culture = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    majority_culture_chi = chi_square_test(majority_by_culture)
    age_bins_social = pd.cut(df_social["age"], bins=[4, 6, 8, 10, 12, 14], include_lowest=True)
    majority_by_agebin = pd.crosstab(age_bins_social, df_social["majority_choice"])
    majority_age_chi = chi_square_test(majority_by_agebin)

    print("\nChi-square majority ~ culture:", majority_culture_chi)
    print("Chi-square majority ~ age_bin:", majority_age_chi)

    # Save a small machine-readable summary in case it's useful downstream
    summary = {
        "n_total": n_total,
        "n_social": n_social,
        "n_majority_social": n_majority,
        "age_p_social": age_p_social,
        "culture_social": culture_eff_social,
        "social_culture_chi": social_culture_chi,
        "social_age_chi": social_age_chi,
        "age_p_majority": age_p_majority,
        "culture_majority": culture_eff_majority,
        "majority_culture_chi": majority_culture_chi,
        "majority_age_chi": majority_age_chi,
    }

    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote analysis_summary.json")


if __name__ == "__main__":
    main()

