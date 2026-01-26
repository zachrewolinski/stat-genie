def extract_final_answer(model_output):
    """
    Extracts the estimated effect of SkinDark on redCards from the provided model_output.
    Returns a dictionary with keys:
      - "object": dict with coefficient, SE, z, p-value, IRR, IRR 95% CI, and a boolean 'significant'
      - "description": brief interpretation of the result in context

    The function handles the clustered_result object (preferred) and falls back to model_result if necessary.
    """
    import numpy as np
    from scipy.stats import norm

    # Helper to format numeric values
    def _flt(x):
        try:
            return float(np.asarray(x))
        except Exception:
            return None

    # Try to use clustered results if available
    clustered = model_output.get('clustered_result', None)
    base = model_output.get('model_result', None)

    if clustered is None and base is None:
        raise ValueError("model_output must contain at least 'clustered_result' or 'model_result'.")

    # Extract params series
    if clustered is not None:
        params = clustered.params
        # Extract standard errors: clustered.bse may be an array (no index) or a Series-like object
        try:
            bse_all = clustered.bse
        except Exception:
            bse_all = None
        # Extract conf_int if available
        try:
            conf_df = clustered.conf_int()
        except Exception:
            conf_df = None
    else:
        # Fallback to model_result
        params = base.params
        cov = base.cov_params()
        bse_all = np.sqrt(np.diag(cov))
        conf_df = None

    if 'SkinDark' not in params.index:
        raise KeyError("SkinDark not found among model parameters.")

    coef = _flt(params['SkinDark'])

    # Determine SE for SkinDark
    se = None
    if bse_all is not None:
        # If bse_all is numpy array, align by position; otherwise try dict-like access
        if isinstance(bse_all, np.ndarray):
            pos = params.index.get_loc('SkinDark')
            se = _flt(bse_all[pos])
        else:
            # could be pd.Series or dict-like
            try:
                se = _flt(bse_all['SkinDark'])
            except Exception:
                # try aligning by index name positions if possible
                try:
                    pos = params.index.get_loc('SkinDark')
                    se = _flt(np.asarray(bse_all)[pos])
                except Exception:
                    se = None

    # If conf_df provided (clustered.conf_int()), use it to get CI on coef
    if conf_df is not None and 'SkinDark' in conf_df.index:
        ci_low = _flt(conf_df.loc['SkinDark', 0])
        ci_high = _flt(conf_df.loc['SkinDark', 1])
    else:
        # compute CI from coef +/- 1.96*SE if SE available
        if se is not None:
            zval = norm.ppf(0.975)
            ci_low = coef - zval * se
            ci_high = coef + zval * se
        else:
            ci_low = ci_high = None

    # Compute z-stat and p-value if SE is available
    if se is not None and se > 0:
        z = coef / se
        p_value = 2 * (1 - norm.cdf(abs(z)))
    else:
        z = None
        p_value = None

    # Compute IRR and IRR CI if possible
    irr = None
    irr_ci_lower = None
    irr_ci_upper = None
    try:
        irr = float(np.exp(coef)) if coef is not None else None
    except Exception:
        irr = None
    if ci_low is not None and ci_high is not None:
        try:
            irr_ci_lower = float(np.exp(ci_low))
            irr_ci_upper = float(np.exp(ci_high))
        except Exception:
            irr_ci_lower = irr_ci_upper = None

    significant = (p_value is not None) and (p_value < 0.05)

    result_object = {
        'coef_log_rate_ratio': coef,
        'se': se,
        'z': z,
        'p_value': p_value,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper,
        'significant_at_0.05': significant,
    }

    # Build description
    if p_value is None:
        desc = "Could not compute p-value/SE for SkinDark from the provided model output, but extracted coefficient and (if available) confidence interval."
    else:
        # Interpret direction and significance
        if significant:
            direction = "higher" if irr is None or irr > 1 else ("higher" if irr > 1 else "lower")
            desc = (
                f"Estimated incidence rate ratio (IRR) for SkinDark = {irr:.3f} "
                f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]). "
                f"Coefficient (log IRR) = {coef:.3f}, SE = {se:.3f}, z = {z:.3f}, p = {p_value:.3f}. "
                f"This indicates that, conditional on controls and clustering by referee, "
                f"players categorized as dark-skinned receive red cards at a statistically significantly "
                f"{direction} rate compared with light-skinned players (alpha=0.05)."
            )
        else:
            # not significant
            if irr is not None:
                desc = (
                    f"Estimated IRR for SkinDark = {irr:.3f} "
                    f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]). "
                    f"Coefficient (log IRR) = {coef:.3f}, SE = {se:.3f}, z = {z:.3f}, p = {p_value:.3f}. "
                    f"The association is not statistically significant at alpha=0.05; "
                    f"evidence is insufficient to conclude that dark-skinned players are more likely to receive red cards."
                )
            else:
                desc = (
                    f"Coefficient (log IRR) for SkinDark = {coef:.3f}. "
                    f"Could not compute full inference (SE/p-value) from the provided output."
                )

    return {'object': result_object, 'description': desc}