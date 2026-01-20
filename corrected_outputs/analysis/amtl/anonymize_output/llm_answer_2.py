def extract_final_answer(model_output):
    """
    Extract the effect of 'IsHuman' from a fitted statsmodels GLMResultsWrapper.

    Returns a dict with:
      - "object": dict containing coefficient, se, z, p-value, 95% CI, odds ratio and OR CI,
                  and a boolean 'significant' (alpha=0.05).
      - "description": short plain-language interpretation of the IsHuman effect
                       in the context of AMTL (antemortem tooth loss).

    Raises a ValueError if 'IsHuman' is not found in the model results.
    """
    import numpy as np
    import pandas as pd

    # Ensure params/index exist
    try:
        param_index = list(model_output.params.index)
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a statsmodels results object with params.") from e

    if 'IsHuman' not in param_index:
        raise ValueError("The model results do not contain a parameter named 'IsHuman'.")

    # Extract point estimate and standard error
    coef = float(model_output.params['IsHuman'])
    # Some results objects provide .bse as Series
    se = float(model_output.bse['IsHuman'])

    # z (or Wald) statistic and p-value
    z_stat = coef / se if se != 0 else np.nan
    # Try to get p-value directly; fall back to normal approximation if not present
    try:
        p_value = float(model_output.pvalues['IsHuman'])
    except Exception:
        # two-sided p-value from normal distribution
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    # 95% confidence interval for coefficient
    try:
        ci = model_output.conf_int()
        # conf_int may be DataFrame or ndarray; handle both
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_lower, ci_upper = [float(x) for x in ci.loc['IsHuman']]
        else:
            # assume numpy array in same order as params
            pos = param_index.index('IsHuman')
            ci_lower, ci_upper = float(ci[pos, 0]), float(ci[pos, 1])
    except Exception:
        # fallback approximate CI using normal approximation
        z_crit = 1.96
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se

    # Odds ratio and its CI (exponentiated)
    or_est = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Significance flag (alpha = 0.05)
    significant = (p_value < 0.05) if (p_value is not None and not np.isnan(p_value)) else False

    # Short conclusion: positive coef => higher odds of AMTL for humans
    if np.isnan(coef):
        conclusion = "Could not determine effect (coefficient is NaN)."
    else:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        sig_text = "statistically significant" if significant else "not statistically significant"
        conclusion = (f"Modern humans (IsHuman=1) have {direction} odds of antemortem tooth loss "
                      f"compared to the non-human primates (IsHuman=0). The effect is {sig_text} "
                      f"(coef={coef:.4f}, OR={or_est:.3f}, p={p_value:.3g}).")

    result_object = {
        "coef": coef,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": or_est,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "significant": significant
    }

    description = (
        "Estimated effect of IsHuman (1 = modern human, 0 = non-human primate) on the log-odds of a tooth being "
        "missing (AMTL). Positive coefficient means higher odds of missing teeth in modern humans. "
        f"Interpretation: {conclusion}"
    )

    return {"object": result_object, "description": description}