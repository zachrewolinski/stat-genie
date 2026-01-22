def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, odds ratios and a short interpretation
    for the key predictors: 'relative_size', 'focal_home_bin', and their interaction
    'relative_size_x_home' from the provided model_output (which may be a statsmodels-like
    results object or the lightweight ClusteredResultsWrapper used in the modeling code).
    
    Returns:
      {
        "object": {
          "relative_size": {coef, se, p, ci_lower, ci_upper, or, or_ci_lower, or_ci_upper, significant},
          "focal_home_bin": {...},
          "relative_size_x_home": {...},
          "raw": {params, bse, pvalues, conf_int (as array/dataframe) }  # for inspection
        },
        "description": "<plain-language summary of effects and significance>"
      }
    """
    import numpy as np
    import math

    # Helper to safely get an attribute (try several common names)
    def _get_attr(obj, *names, default=None):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    # Try to obtain parameter estimates
    params = _get_attr(model_output, 'params', 'parameters')
    # p-values
    pvalues = _get_attr(model_output, 'pvalues', 'pvals', default=None)
    # standard errors
    bse = _get_attr(model_output, 'bse', default=None)
    # conf int: try callable method first (statsmodels style), then attribute
    conf_int = None
    if hasattr(model_output, 'conf_int') and callable(model_output.conf_int):
        try:
            conf_int = model_output.conf_int()
        except Exception:
            conf_int = _get_attr(model_output, 'conf_int', default=None)
    else:
        conf_int = _get_attr(model_output, 'conf_int', default=None)

    # Determine variable names / ordering
    names = None
    if params is None:
        raise ValueError("Model output does not expose parameter estimates ('params').")
    # If params is a pandas Series or has an index, use that
    try:
        # pandas Series or similar
        names = list(params.index)
    except Exception:
        # try model metadata (statsmodels)
        model_exog_names = _get_attr(model_output, 'model', default=None)
        if model_exog_names is not None and hasattr(model_output.model, 'exog_names'):
            names = list(model_output.model.exog_names)
        else:
            # fallback: if params is an ndarray, create generic names based on length
            try:
                params = np.asarray(params)
                names = [f'param_{i}' for i in range(len(params))]
                # wrap params in a dict-like mapping for indexing convenience later
                params = dict(zip(names, params))
            except Exception:
                raise ValueError("Unable to determine parameter names/index from model output.")

    # Normalize params, bse, pvalues, conf_int into dicts keyed by variable names
    def array_to_dict(arr, names):
        if arr is None:
            return {n: None for n in names}
        # If arr is a pandas Series with index, convert to dict
        try:
            return {str(k): float(v) for k, v in arr.items()}
        except Exception:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return {names[i]: float(arr[i]) for i in range(len(names))}
            else:
                # Not 1D: return None mapping
                return {n: None for n in names}

    # params may already be dict-like or Series; handle both
    if isinstance(params, dict):
        params_dict = {str(k): float(v) for k, v in params.items()}
    else:
        params_dict = array_to_dict(params, names)

    bse_dict = array_to_dict(bse, names)
    pval_dict = array_to_dict(pvalues, names)

    # conf_int can be a 2-col array where rows correspond to names order, or a DataFrame
    conf_dict = {}
    if conf_int is None:
        conf_dict = {n: (None, None) for n in names}
    else:
        try:
            # If conf_int is a DataFrame-like with index and two columns
            try:
                # pandas DataFrame case
                lo = conf_int.iloc[:, 0]
                hi = conf_int.iloc[:, 1]
                conf_dict = {str(idx): (float(lo.loc[idx]), float(hi.loc[idx])) for idx in conf_int.index}
            except Exception:
                # array-like: rows correspond to names order
                arr = np.asarray(conf_int)
                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] == len(names):
                    conf_dict = {names[i]: (float(arr[i, 0]), float(arr[i, 1])) for i in range(len(names))}
                else:
                    # unexpected shape
                    conf_dict = {n: (None, None) for n in names}
        except Exception:
            conf_dict = {n: (None, None) for n in names}

    # Variables of primary interest
    target_vars = ['relative_size', 'focal_home_bin', 'relative_size_x_home']

    extracted = {}
    for var in target_vars:
        if var in params_dict:
            coef = params_dict[var]
            se = bse_dict.get(var, None)
            p = pval_dict.get(var, None)
            ci_lo, ci_hi = conf_dict.get(var, (None, None))
            # Odds ratio and CI if coef available
            try:
                or_val = float(math.exp(coef))
            except Exception:
                or_val = None
            try:
                or_ci_lo = float(math.exp(ci_lo)) if ci_lo is not None else None
                or_ci_hi = float(math.exp(ci_hi)) if ci_hi is not None else None
            except Exception:
                or_ci_lo = or_ci_hi = None
            significant = (p is not None) and (p < 0.05)
            extracted[var] = {
                'coef': coef,
                'std_err': se,
                'p_value': p,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'odds_ratio': or_val,
                'or_ci_lower': or_ci_lo,
                'or_ci_upper': or_ci_hi,
                'significant_0.05': bool(significant)
            }
        else:
            extracted[var] = None  # variable not present in model

    # Add the raw parameter tables to the output for inspection
    raw = {
        'params': params_dict,
        'bse': bse_dict,
        'pvalues': pval_dict,
        'conf_int': conf_dict
    }

    # Build a short plain-language description
    lines = []
    # relative_size
    rs = extracted.get('relative_size')
    if rs:
        if rs['p_value'] is not None:
            sigtext = "statistically significant (p < 0.05)" if rs['significant_0.05'] else f"not statistically significant (p = {rs['p_value']:.3g})"
        else:
            sigtext = "p-value not available"
        lines.append(
            f"Relative group size: coef = {rs['coef']:.3g}, OR = {rs['odds_ratio']:.3g} "
            f"(95% CI OR: {rs['or_ci_lower']:.3g}–{rs['or_ci_upper']:.3g}) — {sigtext}."
            if (rs['odds_ratio'] is not None and rs['or_ci_lower'] is not None) else
            f"Relative group size: coef = {rs['coef']:.3g} — {sigtext}."
        )
    else:
        lines.append("Relative group size: not included in model output.")

    # focal_home_bin
    fh = extracted.get('focal_home_bin')
    if fh:
        if fh['p_value'] is not None:
            sigtext = "statistically significant (p < 0.05)" if fh['significant_0.05'] else f"not statistically significant (p = {fh['p_value']:.3g})"
        else:
            sigtext = "p-value not available"
        lines.append(
            f"Focal home location (home advantage): coef = {fh['coef']:.3g}, OR = {fh['odds_ratio']:.3g} "
            f"(95% CI OR: {fh['or_ci_lower']:.3g}–{fh['or_ci_upper']:.3g}) — {sigtext}."
            if (fh['odds_ratio'] is not None and fh['or_ci_lower'] is not None) else
            f"Focal home location: coef = {fh['coef']:.3g} — {sigtext}."
        )
    else:
        lines.append("Focal home location: not included in model output.")

    # interaction
    inter = extracted.get('relative_size_x_home')
    if inter:
        if inter['p_value'] is not None:
            sigtext = "statistically significant (p < 0.05)" if inter['significant_0.05'] else f"not statistically significant (p = {inter['p_value']:.3g})"
        else:
            sigtext = "p-value not available"
        # interpret interaction direction
        direction = ""
        try:
            if inter['coef'] is not None:
                if inter['coef'] > 0:
                    direction = "A positive interaction indicates that the benefit of larger relative size on winning is larger when the focal group is at home."
                elif inter['coef'] < 0:
                    direction = "A negative interaction indicates that the benefit of larger relative size on winning is smaller (or reverses) when the focal group is at home."
        except Exception:
            direction = ""
        lines.append(
            f"Interaction (relative_size x focal_home_bin): coef = {inter['coef']:.3g}, OR = {inter['odds_ratio']:.3g} — {sigtext}. {direction}"
            if (inter['odds_ratio'] is not None) else
            f"Interaction (relative_size x focal_home_bin): coef = {inter['coef']:.3g} — {sigtext}. {direction}"
        )
    else:
        lines.append("Interaction (relative_size x focal_home_bin): not included in model output.")

    description = " ".join(lines)

    return {
        "object": {
            "estimates": extracted,
            "raw": raw
        },
        "description": description
    }