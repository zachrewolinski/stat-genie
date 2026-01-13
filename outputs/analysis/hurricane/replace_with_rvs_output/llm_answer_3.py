def extract_final_answer(model_output):
    """
    Extracts statistics for the femininity predictors from the provided model_output dict.
    Expects model_output to be a dict with keys:
      - 'primary_model' -> statsmodels RegressionResultsWrapper (uses 'masfem_z')
      - 'sensitivity_mturk_model' -> statsmodels RegressionResultsWrapper (uses 'masfem_mturk_z') or None
      - 'deaths_model' -> statsmodels RegressionResultsWrapper (uses 'masfem_z' on log fatalities) or None

    Returns a dict with:
      - "object": dict with extracted numeric stats for each relevant model (estimate, se, t, p, 95% CI, nobs)
      - "description": brief textual interpretation of those stats in relation to the hypothesis:
          "More feminine names -> less precaution -> more logged damage" (prediction: positive effect on log_ndam15).
    """

    def _extract(model, varname):
        """Return stats for varname from a statsmodels fitted model or None if not available."""
        if model is None:
            return None
        try:
            params = model.params
        except Exception:
            return None
        # params might be a Series with index
        try:
            if varname not in params.index:
                return None
        except Exception:
            return None

        # Extract core statistics with safe access
        try:
            est = float(params[varname])
        except Exception:
            est = None
        try:
            se = float(model.bse[varname]) if hasattr(model, "bse") and varname in model.bse.index else None
        except Exception:
            se = None
        try:
            t = float(model.tvalues[varname]) if hasattr(model, "tvalues") and varname in model.tvalues.index else None
        except Exception:
            t = None
        try:
            p = float(model.pvalues[varname]) if hasattr(model, "pvalues") and varname in model.pvalues.index else None
        except Exception:
            p = None

        # confidence interval extraction (robust HC3 used when fitting)
        ci_lower = ci_upper = None
        try:
            ci = model.conf_int()
            try:
                # DataFrame with index
                row = ci.loc[varname]
                ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
            except Exception:
                # fallback if conf_int returns ndarray or has no loc
                try:
                    idx = list(params.index).index(varname)
                    row = ci[idx]
                    ci_lower, ci_upper = float(row[0]), float(row[1])
                except Exception:
                    ci_lower = ci_upper = None
        except Exception:
            ci_lower = ci_upper = None

        # number of observations
        try:
            nobs = int(model.nobs)
        except Exception:
            # fallback: infer from model.model.endog
            try:
                nobs = int(model.model.endog.shape[0])
            except Exception:
                nobs = None

        # Decide whether the point estimate supports the directional hypothesis (positive effect)
        supports_direction = None
        if est is not None:
            supports_direction = (est > 0)

        # Decide whether the result is statistically "significant" at p < 0.05 (if p available)
        significant = None
        if p is not None:
            significant = (p < 0.05)

        return {
            'variable': varname,
            'estimate': est,
            'std_error': se,
            't_value': t,
            'p_value': p,
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
            'nobs': nobs,
            # boolean indicators
            'supports_directional_hypothesis (est>0)': supports_direction,
            'statistically_significant_at_0.05': significant,
            # also provide short keys used in textual summaries
            'dir': supports_direction,
            'sig': significant
        }

    # Safely get models from input dict-like object
    try:
        primary_model = model_output.get('primary_model')
    except Exception:
        primary_model = None
    try:
        sens_model = model_output.get('sensitivity_mturk_model')
    except Exception:
        sens_model = None
    try:
        deaths_model = model_output.get('deaths_model')
    except Exception:
        deaths_model = None

    primary_stats = _extract(primary_model, 'masfem_z')
    sens_stats = _extract(sens_model, 'masfem_mturk_z')
    deaths_stats = _extract(deaths_model, 'masfem_z')  # same IV but outcome = log fatalities

    # Helper to create display strings safely
    def _fmt_num(val, fmt):
        if val is None:
            return 'NA'
        try:
            return fmt.format(val)
        except Exception:
            try:
                return fmt.format(float(val))
            except Exception:
                return 'NA'

    parts = []
    if primary_stats is not None:
        primary_display = {
            'estimate': _fmt_num(primary_stats.get('estimate'), "{:.4f}"),
            'std_error': _fmt_num(primary_stats.get('std_error'), "{:.4f}"),
            't_value': _fmt_num(primary_stats.get('t_value'), "{:.2f}"),
            'p_value': _fmt_num(primary_stats.get('p_value'), "{:.3g}"),
            'ci_lower_95': _fmt_num(primary_stats.get('ci_lower_95'), "{:.4f}"),
            'ci_upper_95': _fmt_num(primary_stats.get('ci_upper_95'), "{:.4f}"),
            'nobs': _fmt_num(primary_stats.get('nobs'), "{}"),
            'dir': str(primary_stats.get('dir')) if primary_stats.get('dir') is not None else 'NA',
            'sig': str(primary_stats.get('sig')) if primary_stats.get('sig') is not None else 'NA'
        }
        parts.append(
            "Primary model (DV = log property damage): estimate for masfem_z = {estimate}, SE = {std_error}, "
            "t = {t_value}, p = {p_value}, 95% CI = [{ci_lower_95}, {ci_upper_95}], n = {nobs}. "
            "Estimate > 0: {dir}, significant (p<0.05): {sig}.".format(**primary_display)
        )
    else:
        parts.append("Primary model: masfem_z not available or extraction failed.")

    if sens_stats is not None:
        sens_display = {
            'estimate': _fmt_num(sens_stats.get('estimate'), "{:.4f}"),
            'std_error': _fmt_num(sens_stats.get('std_error'), "{:.4f}"),
            't_value': _fmt_num(sens_stats.get('t_value'), "{:.2f}"),
            'p_value': _fmt_num(sens_stats.get('p_value'), "{:.3g}"),
            'ci_lower_95': _fmt_num(sens_stats.get('ci_lower_95'), "{:.4f}"),
            'ci_upper_95': _fmt_num(sens_stats.get('ci_upper_95'), "{:.4f}"),
            'nobs': _fmt_num(sens_stats.get('nobs'), "{}"),
            'dir': str(sens_stats.get('dir')) if sens_stats.get('dir') is not None else 'NA',
            'sig': str(sens_stats.get('sig')) if sens_stats.get('sig') is not None else 'NA'
        }
        parts.append(
            "Sensitivity (MTurk femininity): estimate for masfem_mturk_z = {estimate}, SE = {std_error}, "
            "t = {t_value}, p = {p_value}, 95% CI = [{ci_lower_95}, {ci_upper_95}], n = {nobs}. "
            "Estimate > 0: {dir}, significant (p<0.05): {sig}.".format(**sens_display)
        )
    else:
        parts.append("Sensitivity (MTurk) model: masfem_mturk_z not available or extraction failed/absent.")

    if deaths_stats is not None:
        deaths_display = {
            'estimate': _fmt_num(deaths_stats.get('estimate'), "{:.4f}"),
            'std_error': _fmt_num(deaths_stats.get('std_error'), "{:.4f}"),
            't_value': _fmt_num(deaths_stats.get('t_value'), "{:.2f}"),
            'p_value': _fmt_num(deaths_stats.get('p_value'), "{:.3g}"),
            'ci_lower_95': _fmt_num(deaths_stats.get('ci_lower_95'), "{:.4f}"),
            'ci_upper_95': _fmt_num(deaths_stats.get('ci_upper_95'), "{:.4f}"),
            'nobs': _fmt_num(deaths_stats.get('nobs'), "{}"),
            'dir': str(deaths_stats.get('dir')) if deaths_stats.get('dir') is not None else 'NA',
            'sig': str(deaths_stats.get('sig')) if deaths_stats.get('sig') is not None else 'NA'
        }
        parts.append(
            "Deaths model (DV = log fatalities): estimate for masfem_z = {estimate}, SE = {std_error}, "
            "t = {t_value}, p = {p_value}, 95% CI = [{ci_lower_95}, {ci_upper_95}], n = {nobs}. "
            "Estimate > 0: {dir}, significant (p<0.05): {sig}.".format(**deaths_display)
        )
    else:
        parts.append("Deaths model: masfem_z on fatalities not available or extraction failed/absent.")

    interpretation = " ".join(parts)

    # Final verdict about hypothesis: require positive estimate AND p<0.05 in primary model to claim support.
    final_verdict = None
    if primary_stats is None:
        final_verdict = "Cannot determine: primary model stats unavailable."
    else:
        est = primary_stats.get('estimate')
        pval = primary_stats.get('p_value')
        if est is not None and pval is not None:
            if est > 0 and pval < 0.05:
                final_verdict = ("Primary model provides statistically significant evidence (p < 0.05) that more "
                                 "feminine names are associated with higher logged property damage, "
                                 "which is consistent with the hypothesis (less precaution => more damage).")
            elif est > 0:
                final_verdict = ("Primary model estimate is positive (consistent directionally with the hypothesis) "
                                 "but not statistically significant at p < 0.05.")
            else:
                final_verdict = ("Primary model estimate is not positive (or is zero) and therefore does not support the "
                                 "hypothesized direction.")
        else:
            final_verdict = "Primary model: insufficient information to form a statistical verdict."

    # Build result object (keep the numeric dictionaries intact)
    result_object = {
        'primary_model_stats': primary_stats,
        'sensitivity_mturk_stats': sens_stats,
        'deaths_model_stats': deaths_stats,
        'final_verdict': final_verdict
    }

    description = ("Extracted coefficients, SEs, t-values, p-values, 95% CIs, and sample sizes for the femininity "
                   "predictors from the primary model (masfem_z), the MTurk-based sensitivity (masfem_mturk_z), "
                   "and the deaths sensitivity (masfem_z on log fatalities). " + final_verdict + " " + interpretation)

    return {
        "object": result_object,
        "description": description
    }