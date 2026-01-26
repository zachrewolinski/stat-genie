def extract_final_answer(model_output):
    """
    Extract and interpret the effect of 'is_human' from the model output returned
    by the provided modeling function.

    Returns a dict with:
      - "object": dict of extracted statistics for 'is_human' (log-odds coef, SE used,
                  z, two-sided p-value, odds ratio, 95% CI for OR, method used for SE)
      - "description": short plain-language interpretation answering whether modern
                       humans have higher AMTL than the non-human genera after
                       controlling for covariates.
    """
    import numpy as np
    from scipy.stats import norm

    # Helper to safely access items
    fit = model_output.get('fit', None)
    robust = model_output.get('robust_results', None)

    # Ensure we can access parameter estimates
    if robust is None and fit is None:
        raise ValueError("model_output must contain at least 'fit' or 'robust_results'.")

    # Preferred: use cluster-robust results for SEs/confidence intervals if available
    use_robust = False
    method_note = ""
    # Try to pull params from robust wrapper if present
    if robust is not None:
        try:
            params = robust.params  # pandas Series
            # check that 'is_human' is present
            if 'is_human' in params.index:
                use_robust = True
        except Exception:
            use_robust = False

    # Fallback to fit (non-robust)
    if use_robust:
        coef = float(robust.params.loc['is_human'])
        # Try robust SE
        try:
            se = float(robust.bse.loc['is_human'])
            if np.isnan(se) or se == 0.0:
                raise ValueError("robust SE is NaN or zero")
            method_note = "cluster-robust SEs (clustered by 'specimen')"
        except Exception:
            # fallback to fit's bse
            if fit is not None and 'is_human' in fit.params.index:
                se = float(fit.bse.loc['is_human'])
                method_note = "non-robust SEs (fallback to model fit bse)"
            else:
                se = float(np.nan)
                method_note = "no SE available"
        # Confidence interval (log-odds)
        try:
            ci_log = robust.conf_int().loc['is_human']  # DataFrame-like with [0]=lower, [1]=upper
            ci_lower_log, ci_upper_log = float(ci_log.iloc[0]), float(ci_log.iloc[1])
        except Exception:
            # fallback to fit.conf_int()
            try:
                ci_log = fit.conf_int().loc['is_human']
                ci_lower_log, ci_upper_log = float(ci_log.iloc[0]), float(ci_log.iloc[1])
                method_note += " (CI from non-robust fit)"
            except Exception:
                ci_lower_log, ci_upper_log = (np.nan, np.nan)
    else:
        # No robust wrapper available: use fit
        if fit is None:
            raise ValueError("No usable fit object found in model_output.")
        if 'is_human' not in fit.params.index:
            raise ValueError("The fitted model does not contain a parameter named 'is_human'.")
        coef = float(fit.params.loc['is_human'])
        se = float(fit.bse.loc['is_human'])
        ci_log = fit.conf_int().loc['is_human']
        ci_lower_log, ci_upper_log = float(ci_log.iloc[0]), float(ci_log.iloc[1])
        method_note = "non-robust SEs and CIs from the original fit"

    # Compute z, two-sided p-value
    if np.isnan(se) or se == 0.0:
        z = np.nan
        p_value = np.nan
    else:
        z = coef / se
        p_value = 2 * (1 - norm.cdf(abs(z)))

    # Odds ratio and CI on OR scale
    try:
        odds_ratio = float(np.exp(coef))
    except Exception:
        odds_ratio = np.nan
    try:
        ci_lower_or = float(np.exp(ci_lower_log)) if not np.isnan(ci_lower_log) else np.nan
        ci_upper_or = float(np.exp(ci_upper_log)) if not np.isnan(ci_upper_log) else np.nan
    except Exception:
        ci_lower_or, ci_upper_or = (np.nan, np.nan)

    # Formulate a concise conclusion relative to the hypothesis:
    # Hypothesis: modern humans have higher AMTL frequency than non-human primates,
    # i.e., is_human coefficient should be > 0 and statistically significant.
    significance = None
    if not np.isnan(p_value):
        significance = (p_value < 0.05)
    if significance is True:
        if coef > 0:
            conclusion = ("Yes — modern humans show significantly higher AMTL after "
                          "controlling for age, sex, and tooth class (coef>0, p < 0.05).")
        else:
            conclusion = ("No — modern humans show significantly lower AMTL after "
                          "controlling for covariates (coef<0, p < 0.05).")
    elif significance is False:
        conclusion = ("No — there is no statistically significant difference in AMTL "
                      "between modern humans and the non-human primates after controlling for covariates "
                      f"(two-sided p = {p_value:.3g}).")
    else:
        conclusion = ("Unable to determine statistical significance (SE or p-value missing).")

    # Assemble object to return
    result_obj = {
        'parameter': 'is_human',
        'coef_logit': coef,
        'se_used': se,
        'se_method': method_note,
        'z_value': z,
        'p_value_two_sided': p_value,
        'odds_ratio': odds_ratio,
        'ci_lower_or': ci_lower_or,
        'ci_upper_or': ci_upper_or,
        'ci_lower_logit': ci_lower_log,
        'ci_upper_logit': ci_upper_log,
        'n_obs': int(getattr(fit, 'nobs', np.nan)) if fit is not None else np.nan
    }

    description = (
        f"Extracted statistics for the predictor 'is_human': log-odds coef = {coef:.4g}, "
        f"SE used = {se:.4g} ({method_note}), z = {np.nan if np.isnan(z) else f'{z:.3g}'}, "
        f"two-sided p = {np.nan if np.isnan(p_value) else f'{p_value:.3g}'}. "
        f"Odds ratio = {np.nan if np.isnan(odds_ratio) else f'{odds_ratio:.4g}'}, "
        f"95% CI for OR = [{ci_lower_or if not np.isnan(ci_lower_or) else 'NA'}, "
        f"{ci_upper_or if not np.isnan(ci_upper_or) else 'NA'}]. "
        f"Conclusion: {conclusion}"
    )

    return {"object": result_obj, "description": description}