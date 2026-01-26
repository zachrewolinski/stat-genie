def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of applicant gender ('female') from the
    model_output produced by the provided `model` function.

    Returns a dictionary with keys:
      - "object": a dict with numeric results (logit coefficient, robust SE, p-value,
                  95% CI, average marginal effect, its SE and 95% CI, and a short
                  conclusion boolean)
      - "description": brief plain-language interpretation of the results.

    The function expects model_output to be a dict with keys:
      - 'logit_result_robust': fitted result wrapper exposing .params, .bse, .pvalues (or delegate)
      - 'marginal_effects_overall': DataFrame (index includes 'female') with 'dy/dx' and 'Std. Err.' etc.
      - 'predictors': list of predictor names (in the same order used to fit the model)
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    if 'logit_result_robust' not in model_output:
        raise ValueError("model_output missing 'logit_result_robust' key.")
    res_robust = model_output['logit_result_robust']

    predictors = model_output.get('predictors', None)
    if predictors is None:
        raise ValueError("model_output missing 'predictors' key listing model predictors.")

    # Parameter names in the fitted result: constant first, then predictors in the provided order
    param_names = ['const'] + list(predictors)

    # Helper to get index of 'female' parameter
    try:
        female_idx = param_names.index('female')
    except ValueError:
        raise ValueError("'female' not found in param names constructed from predictors.")

    # Extract coefficient, robust SE, p-value and build 95% CI using normal approx
    # The wrapper exposes .params and .bse as numpy arrays. If missing, try to delegate.
    params = np.asarray(getattr(res_robust, 'params'))
    bse = np.asarray(getattr(res_robust, 'bse'))
    # pvalues may be present; if not, try to compute from normal approx
    pvalues = getattr(res_robust, 'pvalues', None)
    if pvalues is None:
        # compute z and p using normal approximation
        with np.errstate(divide='ignore', invalid='ignore'):
            zvals = params / bse
        # two-sided p
        try:
            from scipy import stats as _stats
            pvalues = 2.0 * _stats.norm.sf(np.abs(zvals))
        except Exception:
            # fallback approximate using 1.96 threshold only
            pvalues = np.full_like(params, np.nan, dtype=float)

    coef = float(params[female_idx])
    coef_se = float(bse[female_idx]) if bse.size > female_idx else float('nan')
    coef_p = float(pvalues[female_idx]) if hasattr(pvalues, '__len__') else float(pvalues)

    z_crit = 1.96
    coef_ci_low = coef - z_crit * coef_se
    coef_ci_high = coef + z_crit * coef_se

    # Extract average marginal effect (AME) for 'female' if available
    marg_df = model_output.get('marginal_effects_overall', None)
    marg_effect = None
    marg_se = None
    marg_ci_low = None
    marg_ci_high = None
    if marg_df is not None:
        # Try several likely column names for dy/dx and standard error
        # Expect index contains 'female'
        if 'female' not in marg_df.index:
            # try string conversion of index values
            if any(str(i).lower() == 'female' for i in marg_df.index):
                # find the matching index label
                idx_label = next(i for i in marg_df.index if str(i).lower() == 'female')
            else:
                idx_label = None
        else:
            idx_label = 'female'

        if idx_label is not None:
            # column names seen in the provided output: 'dy/dx', 'Std. Err.', 'Conf. Int. Low', 'Cont. Int. Hi.'
            # try common variants
            def _get_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                    # try case-insensitive match
                    for col in df.columns:
                        if str(col).strip().lower() == c.strip().lower():
                            return col
                return None

            dycol = _get_col(marg_df, ['dy/dx', 'dy/dx', 'dy/dx.'])
            secol = _get_col(marg_df, ['Std. Err.', 'Std. Err', 'std err', 'std. err.', 'Std. Error'])
            lowcol = _get_col(marg_df, ['Conf. Int. Low', 'Conf. Int. Low.', 'CI lower', 'ci low'])
            highcol = _get_col(marg_df, ['Cont. Int. Hi.', 'Conf. Int. Hi.', 'CI upper', 'ci high'])

            try:
                if dycol is not None:
                    marg_effect = float(marg_df.loc[idx_label, dycol])
                if secol is not None:
                    marg_se = float(marg_df.loc[idx_label, secol])
                # if CI columns present use them; otherwise compute using normal approx
                if lowcol is not None and highcol is not None:
                    marg_ci_low = float(marg_df.loc[idx_label, lowcol])
                    marg_ci_high = float(marg_df.loc[idx_label, highcol])
                elif (marg_effect is not None) and (marg_se is not None):
                    marg_ci_low = marg_effect - z_crit * marg_se
                    marg_ci_high = marg_effect + z_crit * marg_se
            except Exception:
                # If any lookup failed, leave marginal values as None
                marg_effect = marg_effect or None
                marg_se = marg_se or None
                marg_ci_low = marg_ci_low or None
                marg_ci_high = marg_ci_high or None

    # Formulate a brief conclusion:
    # Use marginal effect if available (more interpretable: probability points); otherwise use logit coef.
    conclusion = ""
    significance = None
    # Decide significance using coefficient p-value if available
    if not np.isnan(coef_p):
        significance = coef_p < 0.05
    else:
        significance = None

    if marg_effect is not None:
        # interpret marginal effect as change in predicted probability (in probability points)
        concl_dir = "increase" if marg_effect > 0 else ("decrease" if marg_effect < 0 else "no change")
        conclusion = (
            f"Adjusted average marginal effect for female = {marg_effect:.4f} (SE={marg_se:.4f}). "
            f"This implies a {marg_effect:.3f} absolute {concl_dir} in approval probability for females vs. males, "
        )
        if marg_ci_low is not None and marg_ci_high is not None:
            conclusion += f"95% CI = [{marg_ci_low:.4f}, {marg_ci_high:.4f}]. "
            if marg_ci_low > 0 or marg_ci_high < 0:
                conclusion += "CI excludes zero → statistically significant at ~5% level. "
            else:
                conclusion += "CI includes zero → not statistically significant at ~5% level. "
        else:
            # fall back to coefficient p-value if present
            if significance is True:
                conclusion += f"(logit p-value = {coef_p:.3g}, significant at 5%). "
            elif significance is False:
                conclusion += f"(logit p-value = {coef_p:.3g}, not significant at 5%). "
            else:
                conclusion += "(no p-value available). "
    else:
        # No marginal effect: rely on logit coefficient
        dir_text = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        conclusion = (
            f"Logit coefficient for female = {coef:.4f} (robust SE={coef_se:.4f}, p={coef_p:.4g}). "
            f"This indicates {dir_text} log-odds of approval for female applicants controlling for covariates. "
            f"95% CI = [{coef_ci_low:.4f}, {coef_ci_high:.4f}]. "
        )
        if significance is True:
            conclusion += "Coefficient is statistically significant at the 5% level."
        elif significance is False:
            conclusion += "Coefficient is not statistically significant at the 5% level."
        else:
            conclusion += "Significance could not be determined."

    # Construct the object to return (numerical)
    result_object = {
        'logit_coef_female': coef,
        'logit_se_female': coef_se,
        'logit_pvalue_female': coef_p,
        'logit_95CI_female': (coef_ci_low, coef_ci_high),
        'marginal_effect_female': marg_effect,
        'marginal_se_female': marg_se,
        'marginal_95CI_female': (marg_ci_low, marg_ci_high),
        'significant_at_5pct': bool(significance) if significance is not None else None,
        # short verdict using marginal effect if present else coefficient sign
        'verdict': (
            "Female applicants are more likely to be approved (positive, statistically significant)"
            if (marg_effect is not None and marg_ci_low is not None and marg_ci_low > 0)
            or (marg_effect is None and significance is True and coef > 0)
            else (
                "Female applicants are less likely to be approved (negative, statistically significant)"
                if (marg_effect is not None and marg_ci_high is not None and marg_ci_high < 0)
                or (marg_effect is None and significance is True and coef < 0)
                else "No statistically robust evidence of a gender effect"
            )
        )
    }

    description = (
        "Extracted statistics for the 'female' indicator from the logistic regression and the "
        "average marginal effects output. The marginal effect (if present) is the preferred "
        "interpretation: it gives the change in predicted probability of mortgage approval for "
        "female vs. male applicants, holding controls fixed. The logit coefficient, robust SE, "
        "p-value and 95% CI are provided as well. The 'verdict' and 'significant_at_5pct' fields "
        "summarize whether there is evidence that gender affects mortgage approval."
    )

    return {"object": result_object, "description": description}