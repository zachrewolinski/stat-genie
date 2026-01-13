def extract_final_answer(model_output):
    """
    Extract key statistics for the femininity name variables from the provided
    model_output dictionary and produce an interpretable summary.

    Expects model_output to contain fitted statsmodels result objects (GLMResultsWrapper)
    under keys like 'nb_cont_model_robust' and 'nb_bin_model_robust' (or fallbacks).
    Returns a dict with keys:
      - "object": dict with extracted numeric results for masfem_z and gender_mf
      - "description": plain-language interpretation of those results in context
    """
    import numpy as np

    def _get_result(key_candidates):
        for k in key_candidates:
            if k in model_output and model_output[k] is not None:
                return model_output[k]
        return None

    # Try common keys used in the model code (robust results preferred)
    cont_res = _get_result(['nb_cont_model_robust', 'nb_cont_model'])
    bin_res = _get_result(['nb_bin_model_robust', 'nb_bin_model'])

    def _extract_for_var(res, varname):
        out = {
            'coef': None,
            'std_err': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'incidence_rate_ratio': None,
            'irr_ci_lower': None,
            'irr_ci_upper': None,
            'n_obs': None,
        }
        if res is None:
            return out

        params = getattr(res, 'params', None)
        if params is None or varname not in params.index:
            return out

        coef = float(params[varname])
        # standard error and p-value
        std_err = float(res.bse[varname]) if (hasattr(res, 'bse') and varname in res.bse.index) else None
        pval = float(res.pvalues[varname]) if (hasattr(res, 'pvalues') and varname in res.pvalues.index) else None

        # confidence interval (robust conf_int may be present)
        try:
            ci = res.conf_int().loc[varname]
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # fallback if conf_int returns ndarray
            try:
                ci_array = res.conf_int()
                idx = list(res.params.index).index(varname)
                ci_lower, ci_upper = float(ci_array[idx, 0]), float(ci_array[idx, 1])
            except Exception:
                ci_lower, ci_upper = None, None

        # incidence rate ratio and its CI
        irr = np.exp(coef) if coef is not None else None
        irr_ci_lower = np.exp(ci_lower) if ci_lower is not None else None
        irr_ci_upper = np.exp(ci_upper) if ci_upper is not None else None

        # n observations
        nobs = getattr(res, 'nobs', None)
        if nobs is None:
            try:
                nobs = int(len(res.model.endog))
            except Exception:
                nobs = None

        out.update({
            'coef': coef,
            'std_err': std_err,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'incidence_rate_ratio': float(irr) if irr is not None else None,
            'irr_ci_lower': float(irr_ci_lower) if irr_ci_lower is not None else None,
            'irr_ci_upper': float(irr_ci_upper) if irr_ci_upper is not None else None,
            'n_obs': int(nobs) if nobs is not None else None,
        })
        return out

    cont_stats = _extract_for_var(cont_res, 'masfem_z')
    bin_stats = _extract_for_var(bin_res, 'gender_mf')

    # Determine whether results support the hypothesis:
    # hypothesis expects positive coefficient (more feminine -> more deaths)
    def _supports_hypothesis(stat):
        if stat['coef'] is None or stat['p_value'] is None:
            return None
        return bool((stat['coef'] > 0) and (stat['p_value'] < 0.05))

    cont_support = _supports_hypothesis(cont_stats)
    bin_support = _supports_hypothesis(bin_stats)

    # Overall judgement: require at least one specification to show a positive, statistically significant effect
    overall_support = any([s is True for s in [cont_support, bin_support]])

    # Compose a succinct description
    description_lines = []
    description_lines.append("Extracted statistics for femininity predictors from negative binomial models (N ≈ {}).".format(
        cont_stats['n_obs'] if cont_stats['n_obs'] is not None else (bin_stats['n_obs'] if bin_stats['n_obs'] is not None else '?')
    ))
    if cont_stats['coef'] is not None:
        description_lines.append(
            "Continuous masfem (masfem_z): coef = {coef:.4f}, SE = {se:.4f}, p = {p:.3f}, 95% CI = [{lo:.4f}, {hi:.4f}]. "
            "IRR = {irr:.3f}, IRR 95% CI = [{irr_lo:.3f}, {irr_hi:.3f}].".format(
                coef=cont_stats['coef'], se=(cont_stats['std_err'] or 0.0), p=(cont_stats['p_value'] or 0.0),
                lo=(cont_stats['ci_lower'] if cont_stats['ci_lower'] is not None else float('nan')),
                hi=(cont_stats['ci_upper'] if cont_stats['ci_upper'] is not None else float('nan')),
                irr=(cont_stats['incidence_rate_ratio'] or float('nan')),
                irr_lo=(cont_stats['irr_ci_lower'] if cont_stats['irr_ci_lower'] is not None else float('nan')),
                irr_hi=(cont_stats['irr_ci_upper'] if cont_stats['irr_ci_upper'] is not None else float('nan')),
            )
        )
        if cont_support:
            description_lines.append("This specification provides statistically significant evidence (p < 0.05) consistent with the hypothesis.")
        else:
            description_lines.append("This specification does NOT provide statistically significant evidence supporting the hypothesis (effect is not significant).")
    else:
        description_lines.append("Continuous masfem result not found in model_output.")

    if bin_stats['coef'] is not None:
        description_lines.append(
            "Binary female-name (gender_mf): coef = {coef:.4f}, SE = {se:.4f}, p = {p:.3f}, 95% CI = [{lo:.4f}, {hi:.4f}]. "
            "IRR = {irr:.3f}, IRR 95% CI = [{irr_lo:.3f}, {irr_hi:.3f}].".format(
                coef=bin_stats['coef'], se=(bin_stats['std_err'] or 0.0), p=(bin_stats['p_value'] or 0.0),
                lo=(bin_stats['ci_lower'] if bin_stats['ci_lower'] is not None else float('nan')),
                hi=(bin_stats['ci_upper'] if bin_stats['ci_upper'] is not None else float('nan')),
                irr=(bin_stats['incidence_rate_ratio'] or float('nan')),
                irr_lo=(bin_stats['irr_ci_lower'] if bin_stats['irr_ci_lower'] is not None else float('nan')),
                irr_hi=(bin_stats['irr_ci_upper'] if bin_stats['irr_ci_upper'] is not None else float('nan')),
            )
        )
        if bin_support:
            description_lines.append("This specification provides statistically significant evidence (p < 0.05) consistent with the hypothesis.")
        else:
            description_lines.append("This specification does NOT provide statistically significant evidence supporting the hypothesis (effect is not significant).")
    else:
        description_lines.append("Binary gender_mf result not found in model_output.")

    if overall_support:
        description_lines.append("Overall conclusion: At least one specification shows a positive, statistically significant association consistent with the hypothesis.")
    else:
        description_lines.append("Overall conclusion: No clear statistical support for the hypothesis — coefficients are positive but not statistically significant in the provided models.")

    description = " ".join(description_lines)

    return {
        "object": {
            "continuous_spec": cont_stats,
            "binary_spec": bin_stats,
            "cont_support_hypothesis": cont_support,
            "bin_support_hypothesis": bin_support,
            "overall_support_hypothesis": overall_support
        },
        "description": description
    }