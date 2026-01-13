def extract_final_answer(model_output):
    """
    Extract key statistics for the independent variables of interest from the fitted models.

    Input:
      model_output: dict returned by the modeling function. Expected keys (prefer robust versions):
        - 'deaths_model_robust' or 'deaths_model_raw' : GLM NegativeBinomial results (preferred robust)
        - optionally 'damage_model_robust' or 'damage_model_raw' : OLS results on log-damage

    Returns:
      dict with keys:
        - "object": a dict containing extracted statistics (coefficients, SEs, p-values,
                    95% CIs, exponentiated effects (IRR or multiplicative change), sample size)
                    for Masfem_z and gender_mf from the fatalities model (and likewise for damage
                    model if present).
        - "description": human-readable interpretation of those statistics in the context of the task.
    """
    import numpy as np

    # Helper to safely get a results object (prefer robust if available)
    def _get_result(output, robust_key, raw_key):
        if robust_key in output and output[robust_key] is not None:
            return output[robust_key]
        return output.get(raw_key, None)

    # Helper: get parameter/variable names for a results object
    def _get_param_names(res):
        # Try several common places for parameter names
        try:
            params = getattr(res, "params", None)
            if params is not None:
                # pandas Series
                if hasattr(params, "index"):
                    return list(params.index)
                # numpy array but model may hold names
            # statsmodels stores exog_names on model
            if hasattr(res, "model"):
                model = res.model
                if hasattr(model, "exog_names") and model.exog_names is not None:
                    return list(model.exog_names)
                if hasattr(model, "data") and hasattr(model.data, "param_names"):
                    try:
                        return list(model.data.param_names)
                    except Exception:
                        pass
            # fallback: try attribute 'names' or 'columns' on params
            if params is not None and hasattr(params, "names"):
                return list(params.names)
        except Exception:
            pass
        # Give up: return empty list
        return []

    # Helper: get a named value from container that may be Series, ndarray, list, or dict
    def _get_named_value(container, name, param_names):
        if container is None:
            return None
        # pandas Series or DataFrame row/col
        if hasattr(container, "get") and not hasattr(container, "index"):
            # dict-like but not pandas Series (rare)
            return container.get(name, None)
        if hasattr(container, "index") and name in container.index:
            return container[name]
        # numpy array or list/tuple
        if isinstance(container, (np.ndarray, list, tuple)):
            if name in param_names:
                idx = param_names.index(name)
                try:
                    return container[idx]
                except Exception:
                    return None
            return None
        # fallback for dict-like
        try:
            return container[name]
        except Exception:
            return None

    res_deaths = _get_result(model_output, 'deaths_model_robust', 'deaths_model_raw')
    if res_deaths is None:
        raise KeyError("No deaths model found in model_output (expected 'deaths_model_robust' or 'deaths_model_raw').")

    # Variables of interest
    ivs = ['Masfem_z', 'gender_mf']

    # Extract parameter table and names
    params = getattr(res_deaths, 'params', None)
    param_names = _get_param_names(res_deaths)

    bse = getattr(res_deaths, 'bse', None)
    pvalues = getattr(res_deaths, 'pvalues', None)
    # conf_int may return DataFrame or ndarray
    ci_raw = None
    try:
        ci_raw = res_deaths.conf_int()
    except Exception:
        ci_raw = None
    # sample size
    nobs = getattr(res_deaths, 'nobs', None)
    # Model family/link info (if GLM)
    fam_name = None
    link_name = None
    try:
        fam = res_deaths.model.family
        fam_name = fam.__class__.__name__
        link_name = fam.link.__class__.__name__ if hasattr(fam, 'link') else None
    except Exception:
        fam_name = None
        link_name = None

    def _get_ci(var, ci_source, param_names_local):
        # return (lower, upper)
        try:
            if ci_source is None:
                return (np.nan, np.nan)
            if hasattr(ci_source, 'loc') and var in getattr(ci_source, 'index', []):
                lower, upper = ci_source.loc[var].tolist()
            elif hasattr(ci_source, '__len__') and isinstance(ci_source, (list, tuple, np.ndarray)):
                if hasattr(ci_source, 'shape') and len(ci_source.shape) == 2:
                    if var in param_names_local:
                        idx = param_names_local.index(var)
                        lower, upper = ci_source[idx].tolist()
                    else:
                        return (np.nan, np.nan)
                else:
                    return (np.nan, np.nan)
            else:
                # try attribute-like access
                try:
                    row = ci_source[var]
                    lower, upper = row.tolist()
                except Exception:
                    return (np.nan, np.nan)
        except Exception:
            return (np.nan, np.nan)
        return float(lower), float(upper)

    out_deaths = {
        'model_family': fam_name,
        'model_link': link_name,
        'nobs': int(nobs) if nobs is not None else None,
        'variables': {}
    }

    for iv in ivs:
        available = iv in param_names
        # also check if params has index containing name
        if not available and hasattr(params, 'index'):
            available = iv in params.index
        if not available:
            out_deaths['variables'][iv] = {'available': False}
            continue

        coef_raw = _get_named_value(params, iv, param_names)
        coef = float(coef_raw) if coef_raw is not None else None

        se_raw = _get_named_value(bse, iv, param_names)
        se = float(se_raw) if se_raw is not None else None

        p_raw = _get_named_value(pvalues, iv, param_names)
        p = float(p_raw) if p_raw is not None else None

        ci_low, ci_high = _get_ci(iv, ci_raw, param_names)

        # If model is a count model with log link (e.g., NegativeBinomial GLM), exponentiate coef to get IRR
        irr = None
        irr_ci = (None, None)
        try:
            if (fam_name is not None and ('NegativeBinomial' in fam_name or 'NegativeBinomial' in str(fam_name))) or (link_name is not None and 'log' in link_name.lower()):
                if coef is not None:
                    irr = float(np.exp(coef))
                    irr_ci = (float(np.exp(ci_low)) if not np.isnan(ci_low) else None,
                              float(np.exp(ci_high)) if not np.isnan(ci_high) else None)
        except Exception:
            irr = None
            irr_ci = (None, None)

        out_deaths['variables'][iv] = {
            'available': True,
            'coef': coef,
            'std_err': se,
            'p_value': p,
            'ci_95': (ci_low, ci_high),
            'exponentiated': irr,           # IRR for count model (multiplicative effect)
            'exponentiated_ci_95': irr_ci,
        }

    # Optionally extract same stats from damage model (if present)
    res_damage = _get_result(model_output, 'damage_model_robust', 'damage_model_raw')
    out_damage = None
    if res_damage is not None:
        params_d = getattr(res_damage, 'params', None)
        param_names_d = _get_param_names(res_damage)

        bse_d = getattr(res_damage, 'bse', None)
        pvalues_d = getattr(res_damage, 'pvalues', None)
        ci_raw_d = None
        try:
            ci_raw_d = res_damage.conf_int()
        except Exception:
            ci_raw_d = None
        nobs_d = getattr(res_damage, 'nobs', None)
        out_damage = {'nobs': int(nobs_d) if nobs_d is not None else None, 'variables': {}}

        def _get_ci_d(var):
            return _get_ci(var, ci_raw_d, param_names_d)

        for iv in ivs:
            available = iv in param_names_d
            if not available and hasattr(params_d, 'index'):
                available = iv in params_d.index
            if not available:
                out_damage['variables'][iv] = {'available': False}
                continue

            coef_raw = _get_named_value(params_d, iv, param_names_d)
            coef = float(coef_raw) if coef_raw is not None else None

            se_raw = _get_named_value(bse_d, iv, param_names_d)
            se = float(se_raw) if se_raw is not None else None

            p_raw = _get_named_value(pvalues_d, iv, param_names_d)
            p = float(p_raw) if p_raw is not None else None

            ci_low, ci_high = _get_ci_d(iv)

            # outcome is log-damage, so exponentiate coef to get multiplicative change in damage
            mult = None
            mult_ci = (None, None)
            try:
                if coef is not None:
                    mult = float(np.exp(coef))
                    mult_ci = (float(np.exp(ci_low)) if not np.isnan(ci_low) else None,
                               float(np.exp(ci_high)) if not np.isnan(ci_high) else None)
            except Exception:
                mult = None
                mult_ci = (None, None)

            out_damage['variables'][iv] = {
                'available': True,
                'coef': coef,
                'std_err': se,
                'p_value': p,
                'ci_95': (ci_low, ci_high),
                'exponentiated': mult,
                'exponentiated_ci_95': mult_ci,
            }

    # Build a concise description summarizing the key result(s)
    def _format_p(p):
        return f"{p:.4f}" if (p is not None) else "NA"

    def _format_float(x, fmt="{:.4f}"):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NA"
            return fmt.format(x)
        except Exception:
            return "NA"

    def _summarize_var(d, varname):
        if varname not in d['variables'] or not d['variables'][varname]['available']:
            return f"{varname}: not present in model."
        v = d['variables'][varname]
        coef = v.get('coef', None)
        p = v.get('p_value', None)
        irr = v.get('exponentiated', None)
        irr_ci = v.get('exponentiated_ci_95', (None, None))
        # Interpret direction: positive coef -> higher expected counts
        direction = "increase" if (coef is not None and coef > 0) else ("decrease" if (coef is not None and coef < 0) else "no change")
        sig = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
        if irr is not None:
            irr_str = _format_float(irr, "{:.3f}")
            irr_ci_low = _format_float(irr_ci[0], "{:.3f}")
            irr_ci_high = _format_float(irr_ci[1], "{:.3f}")
            coef_str = _format_float(coef, "{:.4f}")
            p_str = _format_p(p)
            return (f"{varname}: coef={coef_str}, p={p_str} ({sig}). "
                    f"Exp(coef)={irr_str} (95% CI {irr_ci_low}–{irr_ci_high}), "
                    f"interpreted as a multiplicative {direction} in expected fatalities.")
        else:
            coef_str = _format_float(coef, "{:.4f}")
            p_str = _format_p(p)
            ci_low, ci_high = v.get('ci_95', (None, None))
            ci_low_s = _format_float(ci_low, "{:.4f}")
            ci_high_s = _format_float(ci_high, "{:.4f}")
            return f"{varname}: coef={coef_str}, p={p_str} ({sig}). 95% CI = ({ci_low_s}, {ci_high_s})."

    desc_lines = []
    desc_lines.append("Fatalities model results:")
    desc_lines.append(f" Model family/link: {fam_name}/{link_name}; nobs={out_deaths['nobs']}")
    for iv in ivs:
        desc_lines.append("  - " + _summarize_var(out_deaths, iv))

    if out_damage is not None:
        desc_lines.append("Damage model (OLS on log-damage) results:")
        desc_lines.append(f" nobs={out_damage['nobs']}")
        for iv in ivs:
            v = out_damage['variables'].get(iv, {'available': False})
            if not v['available']:
                desc_lines.append(f"  - {iv}: not present in damage model.")
            else:
                coef = v.get('coef', None)
                p = v.get('p_value', None)
                mult = v.get('exponentiated', None)
                mult_ci = v.get('exponentiated_ci_95', (None, None))
                sig = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
                coef_s = _format_float(coef, "{:.4f}")
                p_s = _format_p(p)
                mult_s = _format_float(mult, "{:.3f}")
                mult_ci_low = _format_float(mult_ci[0], "{:.3f}")
                mult_ci_high = _format_float(mult_ci[1], "{:.3f}")
                desc_lines.append(
                    f"  - {iv}: coef={coef_s}, p={p_s} ({sig}). exp(coef)={mult_s} (95% CI {mult_ci_low}–{mult_ci_high}), interpreted as multiplicative change in damage."
                )

    description = "\n".join(desc_lines)

    result_object = {'deaths_model': out_deaths}
    if out_damage is not None:
        result_object['damage_model'] = out_damage

    return {"object": result_object, "description": description}