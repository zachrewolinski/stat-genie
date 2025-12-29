def extract_final_answer(model_output):
    """
    Extract key statistics about the effects of the predictors (especially livebait and camper)
    on fish_caught/hour from the fitted model stored in model_output.

    Returns a dict with:
      - "object": a dict of numeric results (coefficients, p-values, conf intervals,
                  exponentiated coefficients (rate ratios), example predicted rates/hr)
      - "description": short interpretation of those results in context
    """
    import numpy as np
    from math import exp
    from scipy.stats import norm

    try:
        import pandas as pd
    except Exception:
        pd = None

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function")

    if 'final_model' not in model_output:
        raise ValueError("model_output does not contain 'final_model'")

    result = model_output['final_model']  # statsmodels results wrapper

    # Helpers to robustly get values from Series/DataFrame/dict-like objects
    def _get_from(obj, name, default=None):
        if obj is None:
            return default
        # pandas Series or dict
        try:
            if hasattr(obj, 'get'):
                val = obj.get(name, default)
                if val is not None:
                    return val
        except Exception:
            pass
        # loc (Series/DataFrame)
        try:
            if hasattr(obj, 'loc'):
                return obj.loc[name]
        except Exception:
            pass
        # indexing
        try:
            return obj[name]
        except Exception:
            pass
        # fallback
        return default

    # Extract parameter table
    try:
        params = result.params.copy()
    except Exception:
        # try attribute access fallback
        params = None

    # Ensure params is pandas Series-like or at least has index of names
    if params is None:
        raise ValueError("Could not extract params from final_model")
    if pd is not None and not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            pass

    # try to get standard errors and p-values; compute fallback if missing
    bse = None
    try:
        bse = result.bse.copy()
    except Exception:
        bse = None

    if bse is not None and pd is not None and not isinstance(bse, pd.Series):
        try:
            bse = pd.Series(bse, index=params.index)
        except Exception:
            pass

    pvalues = None
    try:
        pvalues = result.pvalues.copy()
    except Exception:
        pvalues = None

    if pvalues is not None and pd is not None and not isinstance(pvalues, pd.Series):
        try:
            pvalues = pd.Series(pvalues, index=params.index)
        except Exception:
            pass

    # Confidence intervals: normalize to a dict {name: (lower, upper)}
    conf = None
    try:
        conf_df = result.conf_int().copy()
        if pd is not None and hasattr(conf_df, 'loc'):
            conf = {name: (float(conf_df.loc[name][0]), float(conf_df.loc[name][1])) for name in params.index}
        else:
            # assume array-like aligned with params.index
            conf = {name: (float(conf_df[i, 0]), float(conf_df[i, 1])) for i, name in enumerate(params.index)}
    except Exception:
        # fallback: approximate using normal approximation if bse available
        if bse is not None:
            z = norm.ppf(0.975)
            try:
                lower = params - z * bse
                upper = params + z * bse
                conf = {name: (float(_get_from(lower, name)), float(_get_from(upper, name))) for name in params.index}
            except Exception:
                conf = None
        else:
            conf = None

    # If p-values not present, compute from z = coef / se
    if pvalues is None and bse is not None:
        try:
            z = params / bse
            pvals_arr = 2 * (1 - norm.cdf(np.abs(z)))
            if pd is not None:
                pvalues = pd.Series(pvals_arr, index=params.index)
            else:
                pvalues = {name: float(pvals_arr[i]) for i, name in enumerate(params.index)}
        except Exception:
            pvalues = None

    # Prepare exponentiated coefficients (rate ratios) and their CIs if available
    rr = {}
    rr_ci = {}
    for name in params.index:
        coef = float(_get_from(params, name, 0.0))
        rr[name] = float(np.exp(coef))
        if conf is not None:
            try:
                lower, upper = conf[name]
                rr_ci[name] = (float(np.exp(lower)), float(np.exp(upper)))
            except Exception:
                rr_ci[name] = None
        else:
            rr_ci[name] = None

    # Determine significance at alpha=0.05 for key predictors
    significance = {}
    for name in params.index:
        if pvalues is not None:
            try:
                # pvalues can be Series or dict
                pv = _get_from(pvalues, name, np.nan)
                significance[name] = bool(np.isfinite(pv) and (pv < 0.05))
            except Exception:
                significance[name] = None
        else:
            significance[name] = None

    # Compute some example predicted rates (fish per hour) for interpretable scenarios.
    # Intercept represents log(rate) when all predictors = 0 (and offset handled during model fit).
    examples = {}

    def get_param(name, default=0.0):
        return float(_get_from(params, name, default))

    def predict_rate(livebait=0, camper=0, group_size=0):
        try:
            lp = get_param('const', 0.0)
            lp += get_param('livebait', 0.0) * livebait
            lp += get_param('camper', 0.0) * camper
            lp += get_param('group_size', 0.0) * group_size
            return float(np.exp(lp))  # rate per hour
        except Exception:
            return None

    # Baseline when all predictors = 0
    examples['baseline_livebait0_camper0_group_size0'] = predict_rate(0, 0, 0)

    # Typical group: use mean group_size from the original design matrix if available
    mean_group_size = None
    try:
        exog = getattr(result, 'model', None).exog if getattr(result, 'model', None) is not None else None
        names = getattr(result, 'model', None).exog_names if getattr(result, 'model', None) is not None else None
        if exog is not None and names is not None and 'group_size' in names:
            idx = names.index('group_size')
            mean_group_size = float(np.mean(exog[:, idx]))
    except Exception:
        mean_group_size = None

    if mean_group_size is None:
        # fallback to 1 person if mean not available
        mean_group_size = 1.0

    examples[f'avg_group_size_{mean_group_size:.2f}_livebait0_camper0'] = predict_rate(0, 0, mean_group_size)
    examples[f'avg_group_size_{mean_group_size:.2f}_livebait1_camper0'] = predict_rate(1, 0, mean_group_size)
    examples[f'avg_group_size_{mean_group_size:.2f}_livebait0_camper1'] = predict_rate(0, 1, mean_group_size)

    # Prepare std_errors and p_values as plain dicts if available
    std_errors_out = None
    if bse is not None:
        try:
            std_errors_out = {name: float(_get_from(bse, name, np.nan)) for name in params.index}
        except Exception:
            std_errors_out = None

    p_values_out = None
    if pvalues is not None:
        try:
            p_values_out = {name: float(_get_from(pvalues, name, np.nan)) for name in params.index}
        except Exception:
            p_values_out = None

    # conf is already dict name -> (lower, upper) or None
    conf_int_out = conf if conf is not None else None

    # model name extraction: prefer provided final_model_name, else try to infer family name
    model_name = None
    if 'final_model_name' in model_output and model_output.get('final_model_name') is not None:
        model_name = model_output.get('final_model_name')
    else:
        try:
            mod = getattr(result, 'model', None)
            if mod is not None:
                fam = getattr(mod, 'family', None)
                if fam is not None:
                    # family may have attribute 'name' or be represented by its class name
                    model_name = getattr(fam, 'name', None) or getattr(fam, '__class__', None).__name__ or str(fam)
        except Exception:
            model_name = None

    # Package numeric outputs in a plain dict (floats) for JSON-friendly return
    numeric_output = {
        'coefficients': {name: float(get_param(name)) for name in params.index},
        'std_errors': std_errors_out,
        'p_values': p_values_out,
        'conf_int': conf_int_out,
        'rate_ratios': rr,
        'rate_ratio_conf_int': rr_ci,
        'significant_at_0.05': significance,
        'example_predicted_rates_per_hour': examples,
        'model_name': model_name,
        'aic': (float(getattr(result, 'aic', np.nan)) if getattr(result, 'aic', None) is not None else None)
    }

    # Short description/interpretation focusing on livebait and camper
    # Interpret multiplicative effects (rate ratios) and significance
    def interpret_var(name):
        if name not in numeric_output['coefficients']:
            return f"{name}: not in model"
        coef = numeric_output['coefficients'][name]
        rr_val = numeric_output['rate_ratios'].get(name)
        rr_ci_val = numeric_output['rate_ratio_conf_int'].get(name)
        sig = numeric_output['significant_at_0.05'].get(name)
        s = f"{name}: coef={coef:.3f}, rate ratio={rr_val:.3f}" if rr_val is not None else f"{name}: coef={coef:.3f}"
        if rr_ci_val is not None:
            s += f" (95% CI {rr_ci_val[0]:.3f}–{rr_ci_val[1]:.3f})"
        if sig is True:
            s += " — statistically significant (p < 0.05)"
        elif sig is False:
            s += " — not statistically significant (p >= 0.05)"
        else:
            s += " — significance unknown"
        return s

    description_lines = []
    description_lines.append("Key model-derived findings (multiplicative effects on fish caught per hour):")
    # livebait
    description_lines.append(interpret_var('livebait'))
    # camper
    description_lines.append(interpret_var('camper'))
    # group_size
    description_lines.append(interpret_var('group_size'))
    description_lines.append("")  # blank line
    description_lines.append("Examples: predicted fish caught per hour for an average group (see numeric outputs).")
    description = " ".join(description_lines)

    return {"object": numeric_output, "description": description}