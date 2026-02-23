import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def load_data(path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Derived variables
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def lr_test(full_model, reduced_model, label: str) -> None:
    """
    Likelihood-ratio test comparing a full model to a reduced model.
    """
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df = full_model.df_model - reduced_model.df_model
    p_val = stats.chi2.sf(lr_stat, df)
    print(
        f"\nLR test for additional predictors ({label}): "
        f"chi2({int(df)}) = {lr_stat:.3f}, p = {p_val:.4g}"
    )


def summarize_effects(result, age_var: str, culture_var: str, data: pd.DataFrame, outcome_label: str):
    print("\n" + "=" * 80)
    print(f"Outcome: {outcome_label}")
    print(result.summary())

    # Age effect: predicted difference from youngest to oldest
    age_min = data[age_var].min()
    age_max = data[age_var].max()
    base = data.copy()

    def predict_at_age(a: float) -> float:
        base_age = base.copy()
        base_age[age_var] = a
        return result.predict(base_age).mean()

    p_min = predict_at_age(age_min)
    p_max = predict_at_age(age_max)
    print(f"\nPredicted mean probability at age {age_min:.1f}: {p_min:.3f}")
    print(f"Predicted mean probability at age {age_max:.1f}: {p_max:.3f}")
    print(f"Absolute age difference in predicted probability: {abs(p_max - p_min):.3f}")

    # Culture effect: predicted probability by culture at median age
    median_age = data[age_var].median()
    cultures = sorted(data[culture_var].unique())
    culture_probs = {}
    for c in cultures:
        tmp = base.copy()
        tmp[age_var] = median_age
        tmp[culture_var] = c
        culture_probs[c] = result.predict(tmp).mean()

    print("\nPredicted mean probabilities by culture at median age:")
    for c, p in culture_probs.items():
        print(f"  Culture {c}: {p:.3f}")

    if culture_probs:
        max_c = max(culture_probs, key=culture_probs.get)
        min_c = min(culture_probs, key=culture_probs.get)
        diff_c = culture_probs[max_c] - culture_probs[min_c]
        print(
            f"Largest culture difference (culture {max_c} vs {min_c}): "
            f"{diff_c:.3f}"
        )

    # Simple summary of key p-values
    pvalues = result.pvalues
    print("\nKey p-values:")
    if age_var in pvalues:
        print(f"  Age ({age_var}) p-value: {pvalues[age_var]:.4g}")
    # Culture dummies start with 'C(culture)'
    culture_pvals = {k: v for k, v in pvalues.items() if k.startswith("C(culture)")}
    if culture_pvals:
        max_sig = min(culture_pvals.values())
        print(f"  Smallest culture dummy p-value: {max_sig:.4g}")
    gender_p = pvalues.get("gender")
    if gender_p is not None:
        print(f"  Gender p-value: {gender_p:.4g}")
    mf_p = pvalues.get("majority_first")
    if mf_p is not None:
        print(f"  majority_first p-value: {mf_p:.4g}")


def main():
    df = load_data()

    # Model 1: Reliance on social information (any demonstrated option vs undemonstrated)
    formula_social = "social_choice ~ age + C(culture) + gender + majority_first"
    formula_social_reduced = "social_choice ~ gender + majority_first"
    res_social = fit_logit(formula_social, df)
    res_social_reduced = fit_logit(formula_social_reduced, df)
    summarize_effects(
        res_social,
        age_var="age",
        culture_var="culture",
        data=df,
        outcome_label="Reliance on social information (demonstrated vs undemonstrated)",
    )
    lr_test(res_social, res_social_reduced, label="age + culture for social_choice")

    # Model 2: Preference for majority vs all other options
    formula_majority = "majority_choice ~ age + C(culture) + gender + majority_first"
    formula_majority_reduced = "majority_choice ~ gender + majority_first"
    res_majority = fit_logit(formula_majority, df)
    res_majority_reduced = fit_logit(formula_majority_reduced, df)
    summarize_effects(
        res_majority,
        age_var="age",
        culture_var="culture",
        data=df,
        outcome_label="Preference for majority option (majority vs other)",
    )
    lr_test(res_majority, res_majority_reduced, label="age + culture for majority_choice")


if __name__ == "__main__":
    main()
