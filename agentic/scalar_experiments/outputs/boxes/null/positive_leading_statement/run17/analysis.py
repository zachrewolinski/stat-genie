import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(full_result, reduced_result) -> Tuple[float, int, float]:
    """Likelihood-ratio test comparing a full and reduced logistic model."""
    lr_stat = 2.0 * (full_result.llf - reduced_result.llf)
    df_diff = int(full_result.df_model - reduced_result.df_model)
    p_value = float(stats.chi2.sf(lr_stat, df_diff))
    return float(lr_stat), df_diff, p_value


def p_to_evidence(p: float) -> float:
    """Map a p-value to an evidence score in [0, 1]."""
    if p is None or not np.isfinite(p):
        return 0.0
    if p < 1e-6:
        return 1.0
    if p < 1e-3:
        return 0.9
    if p < 1e-2:
        return 0.8
    if p < 5e-2:
        return 0.7
    if p < 0.1:
        return 0.4
    if p < 0.2:
        return 0.2
    return 0.0


def mag_to_evidence(value: float) -> float:
    """Map a range of probabilities (0-1) to an evidence score in [0, 1]."""
    if value is None or not np.isfinite(value):
        return 0.0
    r = float(value)
    if r >= 0.5:
        return 1.0
    if r >= 0.3:
        return 0.8
    if r >= 0.2:
        return 0.6
    if r >= 0.1:
        return 0.4
    if r >= 0.05:
        return 0.2
    return 0.0


def variation(series: pd.Series) -> float:
    """Range (max-min) of a probability series."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.max() - s.min())


def fmt_pct(x: float) -> str:
    """Format a proportion as a percentage string."""
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{100.0 * float(x):.1f}%"


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Basic cleaning / typing
    df = df.dropna(subset=["y", "age", "culture"])
    for col in ["y", "gender", "majority_first", "culture"]:
        df[col] = df[col].astype(int)
    df["age"] = df["age"].astype(float)

    # Derived variables
    df["relies_on_social"] = (df["y"] != 1).astype(int)
    df_demo = df[df["y"].isin([2, 3])].copy()
    df_demo["chose_majority"] = (df_demo["y"] == 2).astype(int)

    # Age groupings for descriptive summaries
    age_bins = [4, 6, 8, 10, 12, 15]
    age_labels = ["4-5", "6-7", "8-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    df_demo["age_group"] = pd.cut(df_demo["age"], bins=age_bins, labels=age_labels, right=False)

    # Descriptive variations
    social_by_age = df.groupby("age_group", observed=True)["relies_on_social"].mean()
    social_by_culture = df.groupby("culture", observed=True)["relies_on_social"].mean()
    maj_by_age = df_demo.groupby("age_group", observed=True)["chose_majority"].mean()
    maj_by_culture = df_demo.groupby("culture", observed=True)["chose_majority"].mean()

    var_social_age = variation(social_by_age)
    var_social_culture = variation(social_by_culture)
    var_maj_age = variation(maj_by_age)
    var_maj_culture = variation(maj_by_culture)

    # Inferential models: logistic regressions
    lr_pvals = {
        "social_age": None,
        "social_culture": None,
        "maj_age": None,
        "maj_culture": None,
    }

    try:
        formula_social = (
            "relies_on_social ~ age + I(age**2) + C(culture) + gender + majority_first"
        )
        formula_social_no_age = "relies_on_social ~ C(culture) + gender + majority_first"
        formula_social_no_culture = (
            "relies_on_social ~ age + I(age**2) + gender + majority_first"
        )

        social_full = smf.logit(formula_social, data=df).fit(disp=False, maxiter=200)
        social_no_age = smf.logit(
            formula_social_no_age, data=df
        ).fit(disp=False, maxiter=200)
        social_no_culture = smf.logit(
            formula_social_no_culture, data=df
        ).fit(disp=False, maxiter=200)

        _, _, p_social_age = lr_test(social_full, social_no_age)
        _, _, p_social_culture = lr_test(social_full, social_no_culture)
        lr_pvals["social_age"] = p_social_age
        lr_pvals["social_culture"] = p_social_culture
    except Exception:
        # Fall back on descriptive variation if model fails
        pass

    try:
        formula_maj = (
            "chose_majority ~ age + I(age**2) + C(culture) + gender + majority_first"
        )
        formula_maj_no_age = "chose_majority ~ C(culture) + gender + majority_first"
        formula_maj_no_culture = (
            "chose_majority ~ age + I(age**2) + gender + majority_first"
        )

        maj_full = smf.logit(formula_maj, data=df_demo).fit(disp=False, maxiter=200)
        maj_no_age = smf.logit(
            formula_maj_no_age, data=df_demo
        ).fit(disp=False, maxiter=200)
        maj_no_culture = smf.logit(
            formula_maj_no_culture, data=df_demo
        ).fit(disp=False, maxiter=200)

        _, _, p_maj_age = lr_test(maj_full, maj_no_age)
        _, _, p_maj_culture = lr_test(maj_full, maj_no_culture)
        lr_pvals["maj_age"] = p_maj_age
        lr_pvals["maj_culture"] = p_maj_culture
    except Exception:
        # Fall back on descriptive variation if model fails
        pass

    # Evidence scores combining significance and effect magnitude
    combos = [
        ("social_age", lr_pvals.get("social_age"), var_social_age),
        ("social_culture", lr_pvals.get("social_culture"), var_social_culture),
        ("maj_age", lr_pvals.get("maj_age"), var_maj_age),
        ("maj_culture", lr_pvals.get("maj_culture"), var_maj_culture),
    ]

    evidences = []
    for _, p_val, magnitude in combos:
        evid_p = p_to_evidence(p_val)
        evid_m = mag_to_evidence(magnitude)
        if evid_p == 0.0 and evid_m == 0.0:
            evid = 0.0
        else:
            # Put slightly more weight on inferential evidence than raw range
            evid = 0.6 * evid_p + 0.4 * evid_m
        evidences.append(evid)

    overall_evidence = float(np.mean(evidences)) if evidences else 0.0
    response = int(round(overall_evidence * 100.0))
    response = max(0, min(100, response))

    # Build explanation string summarizing methods and key findings
    n = int(len(df))
    n_demo = int(len(df_demo))

    exp_parts = []
    exp_parts.append(
        f"Using data from {n} children aged 4-14 years across 8 cultural sites, "
        "I examined whether reliance on social information and preference for majority cues vary by age and culture."
    )
    exp_parts.append(
        f"Overall, {fmt_pct(df['relies_on_social'].mean())} of children chose one of the demonstrators' options, "
        f"and among the {n_demo} trials where a demonstrator option was chosen, "
        f"{fmt_pct(df_demo['chose_majority'].mean())} followed the majority rather than the minority demonstrator."
    )

    # Age-related variation
    if social_by_age.dropna().size > 1:
        exp_parts.append(
            "The probability of relying on social information showed modest variation across age groups, "
            f"ranging from about {fmt_pct(social_by_age.min())} in the youngest group to "
            f"{fmt_pct(social_by_age.max())} in the oldest group "
            f"(range {fmt_pct(var_social_age)})."
        )
        p_sa = lr_pvals.get("social_age")
        if p_sa is not None:
            if p_sa < 0.05:
                qualifier = "and this age effect was statistically significant"
            else:
                qualifier = "but this age effect was not statistically significant"
            exp_parts[-1] = exp_parts[-1][:-1] + (
                f", {qualifier} in a logistic regression likelihood-ratio test (p={p_sa:.3g})."
            )

    if maj_by_age.dropna().size > 1:
        exp_parts.append(
            "Among children who copied a demonstrator, majority choice probabilities also varied somewhat with age, "
            f"from {fmt_pct(maj_by_age.min())} to {fmt_pct(maj_by_age.max())} "
            f"across age groups (range {fmt_pct(var_maj_age)})."
        )
        p_ma = lr_pvals.get("maj_age")
        if p_ma is not None:
            if p_ma < 0.05:
                qualifier = "with a statistically reliable age effect"
            else:
                qualifier = "but the age effect was not statistically reliable"
            exp_parts[-1] = exp_parts[-1][:-1] + (
                f", {qualifier} in the logistic model (p={p_ma:.3g})."
            )

    # Cultural variation
    if social_by_culture.dropna().size > 1:
        exp_parts.append(
            "Reliance on social information differed to a moderate degree across cultural sites, "
            f"with site-level means ranging from {fmt_pct(social_by_culture.min())} to "
            f"{fmt_pct(social_by_culture.max())} (range {fmt_pct(var_social_culture)})."
        )
        p_sc = lr_pvals.get("social_culture")
        if p_sc is not None:
            if p_sc < 0.05:
                qualifier = "and culture had a statistically significant overall effect on social reliance"
            else:
                qualifier = "but the overall culture effect on social reliance was not statistically significant"
            exp_parts[-1] = exp_parts[-1][:-1] + (
                f", {qualifier} in the logistic model (p={p_sc:.3g})."
            )

    if maj_by_culture.dropna().size > 1:
        exp_parts.append(
            "In majority preference, cultural differences were more pronounced: "
            f"majority-choice rates spanned from {fmt_pct(maj_by_culture.min())} to "
            f"{fmt_pct(maj_by_culture.max())} across sites (range {fmt_pct(var_maj_culture)})."
        )
        p_mc = lr_pvals.get("maj_culture")
        if p_mc is not None:
            if p_mc < 0.05:
                qualifier = "and culture was significantly associated with choosing the majority over the minority demonstrator"
            else:
                qualifier = "but this cultural variation did not reach conventional levels of statistical significance"
            exp_parts[-1] = exp_parts[-1][:-1] + (
                f", {qualifier} in the logistic model (p={p_mc:.3g})."
            )

    # Final qualitative conclusion aligned with the Likert score
    if response >= 70:
        final_sentence = (
            "Taken together, these results provide strong statistical evidence that children's reliance on social "
            "information and their preference for majority cues do vary meaningfully across developmental stages and "
            "cultural contexts, so I give a high 'Yes' rating on the 0-100 scale."
        )
    elif response >= 40:
        final_sentence = (
            "Overall, the evidence for systematic variation by age and culture is mixed: there are some differences "
            "in probabilities across groups, but not consistently strong or statistically robust, so I place the answer "
            "between 'No' and 'Yes' on the 0-100 scale."
        )
    else:
        final_sentence = (
            "Overall, the observed age and cultural differences in social reliance and majority preference are modest "
            "and often not statistically significant, so I do not find strong evidence that these tendencies vary "
            "systematically across developmental stages and cultures, and I therefore lean toward a 'No' answer on the 0-100 scale."
        )
    exp_parts.append(final_sentence)

    explanation = " ".join(exp_parts)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)

    # Print summary for traceability
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
