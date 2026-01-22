def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and odds-ratios
    for the predictors of interest from a statsmodels result object contained in model_output.

    Returns a dict with:
      - "object": dict mapping each target variable to its extracted stats
      - "description": brief interpretation of those stats in the context of the question
    """
    import numpy as np

    # Choose clustered_result if present (preferred), otherwise fall back to fit_result
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('clustered_result') or model_output.get('fit_result')
    else:
        res = model_output

    if res is None:
        raise ValueError("No result object found in model_output (expected keys 'clustered_result' or 'fit_result').")

    # Variables of primary interest
    target_vars = ['rel_size_z', 'dist_diff_z', 'rel_size_z:dist_diff_z']

    # Prepare output structure
    stats = {}

    # Attempt to get attributes; handle missing attributes robustly
    params = getattr(res, 'params', None)
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)

    # conf_int is usually a method; try to call if callable
    conf = None
    try:
        maybe_conf = getattr(res, 'conf_int', None)
        if callable(maybe_conf):
            conf = maybe_conf()
        else:
            conf = maybe_conf
    except Exception:
        conf = None

    # Try to obtain parameter names (exog_names) to map numpy arrays to names
    names = None
    try:
        if params is not None:
            if hasattr(params, 'index'):
                # pandas Series / Index
                names = list(params.index)
            elif hasattr(res, 'model') and getattr(res.model, 'exog_names', None) is not None:
                names = list(res.model.exog_names)
            # otherwise leave names as None (we cannot map ndarray without names)
        else:
            if hasattr(res, 'model') and getattr(res.model, 'exog_names', None) is not None:
                names = list(res.model.exog_names)
    except Exception:
        names = None

    def get_named_value(obj, name, names):
        """
        Robustly get a value (or row) for variable `name` from obj which can be:
        - pandas Series (has .loc and .index)
        - pandas DataFrame (has .loc and .index) -> returns a row (Series/ndarray)
        - dict-like
        - numpy array / list (requires `names` to map indices)
        Returns None if not found.
        """
        if obj is None:
            return None

        # dict-like get
        try:
            if isinstance(obj, dict):
                return obj.get(name, None)
        except Exception:
            pass

        # pandas-like .loc / .index access
        try:
            idx = getattr(obj, 'index', None)
            if idx is not None and name in idx:
                try:
                    return obj.loc[name]
                except Exception:
                    # fallback to direct indexing
                    try:
                        return obj[name]
                    except Exception:
                        pass
        except Exception:
            pass

        # If object has a get method (but not dict), try it
        try:
            getm = getattr(obj, 'get', None)
            if callable(getm):
                val = getm(name)
                if val is not None:
                    return val
        except Exception:
            pass

        # If obj is sequence-like (list/tuple/ndarray) use names mapping
        try:
            if isinstance(obj, (list, tuple, np.ndarray)) and names is not None:
                if name in names:
                    idx = names.index(name)
                    try:
                        return obj[idx]
                    except Exception:
                        return None
        except Exception:
            pass

        return None

    def safe_float(x):
        """Convert x to Python float if possible, otherwise return None."""
        if x is None:
            return None
        try:
            # For numpy scalar / pandas scalar / python numeric
            ax = np.asarray(x)
            # If it's an array with shape (), take item()
            if ax.shape == ():
                return float(ax.item())
            # If it's a 1-element array/Series, take first element
            if ax.size == 1:
                return float(ax.flatten()[0])
            # Otherwise cannot convert to single float
            return float(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return None

    for var in target_vars:
        entry = {
            'coef': None,
            'se': None,
            'z_or_t': None,
            'p_value': None,
            'conf_int': None,       # [low, high] on log-odds scale
            'odds_ratio': None,
            'odds_ratio_CI': None   # [low, high] on odds-ratio scale
        }

        raw_coef = get_named_value(params, var, names)
        entry['coef'] = safe_float(raw_coef)

        raw_se = get_named_value(bse, var, names)
        entry['se'] = safe_float(raw_se)

        # compute z (or t) if possible
        if entry['coef'] is not None and entry['se'] is not None:
            if entry['se'] != 0:
                entry['z_or_t'] = float(entry['coef'] / entry['se'])
            else:
                entry['z_or_t'] = None

        raw_pv = get_named_value(pvalues, var, names)
        entry['p_value'] = safe_float(raw_pv)

        raw_conf = get_named_value(conf, var, names)
        # raw_conf may be a sequence-like of length 2, or a pandas Series with two entries
        if raw_conf is not None:
            try:
                arr = np.asarray(raw_conf, dtype=float)
                if arr.size == 2:
                    low, high = float(arr.flatten()[0]), float(arr.flatten()[1])
                    entry['conf_int'] = [low, high]
            except Exception:
                # If conf is a DataFrame and the row access returned something unusual, try more attempts
                try:
                    # If raw_conf has attributes low/high or 0/1 keys
                    if hasattr(raw_conf, '__getitem__'):
                        try:
                            low = safe_float(raw_conf[0])
                            high = safe_float(raw_conf[1])
                            if low is not None and high is not None:
                                entry['conf_int'] = [low, high]
                        except Exception:
                            pass
                except Exception:
                    entry['conf_int'] = None

        # odds ratio and CI (if coef/conf available)
        if entry['coef'] is not None:
            try:
                entry['odds_ratio'] = float(np.exp(entry['coef']))
            except Exception:
                entry['odds_ratio'] = None
        if entry['conf_int'] is not None:
            try:
                entry['odds_ratio_CI'] = [float(np.exp(entry['conf_int'][0])), float(np.exp(entry['conf_int'][1]))]
            except Exception:
                entry['odds_ratio_CI'] = None

        stats[var] = entry

    # Build a concise interpretation/description
    # Use available p-values when present to comment on significance; otherwise note missing SEs/p-values.
    def interpret_var(name, info):
        if info['coef'] is None:
            return f"{name}: coefficient not available."
        text = f"{name}: coefficient = {info['coef']:.3f}"
        if info['se'] is not None:
            text += f", SE = {info['se']:.3f}"
        else:
            text += ", SE = NA"
        if info['p_value'] is not None:
            text += f", p = {info['p_value']:.3f}"
            if info['p_value'] < 0.05:
                sig = "statistically significant (p < 0.05)."
            else:
                sig = "not statistically significant (p >= 0.05)."
            text += f" => {sig}"
        else:
            text += " (no p-value available; inference uncertain)."

        if info['odds_ratio'] is not None:
            text += f" Odds ratio = {info['odds_ratio']:.3f}"
            if info['odds_ratio_CI'] is not None:
                lo, hi = info['odds_ratio_CI']
                text += f" (95% CI: {lo:.3f} to {hi:.3f})"
        return text

    interpretations = [interpret_var(v, stats[v]) for v in target_vars]

    # Overall summary emphasizing practical conclusions and caution about missing SEs/NaNs
    summary_lines = []
    summary_lines.append("Summary interpretation for effects on probability that the focal group wins:")
    # rel_size_z
    rel = stats['rel_size_z']
    if rel['coef'] is not None:
        if rel['p_value'] is not None:
            if rel['p_value'] < 0.05:
                summary_lines.append(
                    f"- Relative group size (rel_size_z) has a positive, statistically significant effect "
                    f"(coef = {rel['coef']:.3f}, p = {rel['p_value']:.3f}). Larger focal groups more likely to win."
                )
            else:
                summary_lines.append(
                    f"- Relative group size (rel_size_z) has a positive effect (coef = {rel['coef']:.3f}) "
                    f"but it is not statistically significant (p = {rel['p_value']:.3f})."
                )
        else:
            summary_lines.append(
                f"- Relative group size (rel_size_z) has a positive coefficient (coef = {rel['coef']:.3f}) "
                "but standard error / p-value are not available (inference not possible)."
            )
    else:
        summary_lines.append("- Relative group size (rel_size_z): coefficient not available.")

    # dist_diff_z
    dist = stats['dist_diff_z']
    if dist['coef'] is not None:
        if dist['p_value'] is not None:
            if dist['p_value'] < 0.05:
                summary_lines.append(
                    f"- Location advantage (dist_diff_z: focal closer to home) is associated with higher win probability "
                    f"(coef = {dist['coef']:.3f}, p = {dist['p_value']:.3f})."
                )
            else:
                summary_lines.append(
                    f"- Location advantage (dist_diff_z) shows a positive coefficient (coef = {dist['coef']:.3f}) "
                    f"but is not statistically significant (p = {dist['p_value']:.3f})."
                )
        else:
            summary_lines.append(
                f"- Location advantage (dist_diff_z) coef = {dist['coef']:.3f}, but SE/p-value not available for formal inference."
            )
    else:
        summary_lines.append("- Location advantage (dist_diff_z): coefficient not available.")

    # interaction
    inter = stats['rel_size_z:dist_diff_z']
    if inter['coef'] is not None:
        if inter['p_value'] is not None:
            if inter['p_value'] < 0.05:
                summary_lines.append(
                    f"- The interaction (rel_size_z:dist_diff_z) is statistically significant (coef = {inter['coef']:.3f}, p = {inter['p_value']:.3f}), "
                    "indicating the effect of relative size depends on location advantage."
                )
            else:
                summary_lines.append(
                    f"- The interaction has a negative coefficient (coef = {inter['coef']:.3f}) but is not statistically significant (p = {inter['p_value']:.3f})."
                )
        else:
            summary_lines.append(
                f"- Interaction coef = {inter['coef']:.3f}, but SE/p-value not available to assess significance."
            )
    else:
        summary_lines.append("- Interaction (rel_size_z:dist_diff_z): coefficient not available.")

    # Note about potential estimation issues seen in the supplied output
    summary_lines.append(
        "Note: Several covariates / fixed effects in the model have missing standard errors or extreme values "
        "in the provided result object. This suggests possible separation, collinearity, or insufficient data for some levels. "
        "Interpret the above estimates with caution; where SE/p-values are missing, formal inference is not possible."
    )

    description = "\n".join(interpretations + [""] + summary_lines)

    return {
        "object": stats,
        "description": description
    }