def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, and multiplicative effect
    (exp(coef)) for predictors of interest from a statsmodels MixedLMResults-like object.
    
    Returns a dictionary with keys:
      - "object": dict containing numeric results for each predictor and some model diagnostics
      - "description": human-readable summary interpreting the extracted statistics
    """
    import numpy as np
    from math import exp
    try:
        from scipy import stats as _scipy_stats
    except Exception:
        _scipy_stats = None

    # Predictors of interest (as used in the model specification)
    predictors = ['age_c', 'sex_m', 'help_y']

    # Prepare container for results
    estimates = {}

    # Extract basic arrays/Series from the fitted model object with safe fallbacks
    params = getattr(model_output, 'params', None)
    bse = getattr(model_output, 'bse', None)
    pvalues = getattr(model_output, 'pvalues', None)
    # Confidence intervals: try method, fallback to params +/- 1.96*bse
    try:
        ci = model_output.conf_int()
    except Exception:
        ci = None

    # If pvalues missing but params and bse present, compute z and p from normal
    if pvalues is None and params is not None and bse is not None:
        z = params / bse
        if _scipy_stats is not None:
            pvalues = 2 * (1 - _scipy_stats.norm.cdf(np.abs(z)))
        else:
            # approximate using math.erf if scipy not available
            from math import erfc, sqrt
            pvalues = 2 * 0.5 * np.array([erfc(abs(zi) / sqrt(2)) for zi in z])

    # Ensure params and bse are present
    if params is None or bse is None:
        raise ValueError("Model output does not expose 'params' and 'bse' attributes needed for extraction.")

    # Convert params/bse/pvalues to pandas-like lookup if they are arrays
    # They are often pandas Series indexed by parameter name; handle both cases.
    def lookup(series_like, name):
        # If it's a pandas Series or DataFrame row-like, try name lookup
        try:
            return series_like[name]
        except Exception:
            # Otherwise assume it's an array and use position-based lookup if possible
            try:
                # find index by exact match in index if available
                idx = list(series_like.index).index(name)
                return series_like[idx]
            except Exception:
                return None

    for pred in predictors:
        coef = lookup(params, pred)
        se = lookup(bse, pred)
        pv = None if pvalues is None else lookup(pvalues, pred)

        # If any missing by name, try partial match (e.g., formula-created names)
        if coef is None:
            # try names that end with the predictor (common when using patsy)
            for n in list(getattr(params, 'index', []) or []):
                if n.endswith(pred):
                    coef = params[n]
                    se = bse[n]
                    pv = None if pvalues is None else pvalues[n]
                    break

        if coef is None:
            # Leave an informative None entry
            estimates[pred] = {
                'coef': None,
                'se': None,
                'pvalue': None,
                'ci_lower': None,
                'ci_upper': None,
                'exp_coef': None,
                'exp_ci_lower': None,
                'exp_ci_upper': None,
                'significant': None,
                'note': f"Predictor '{pred}' not found in model parameters."
            }
            continue

        # Confidence intervals
        if ci is not None:
            try:
                ci_row = ci.loc[pred]
                ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
            except Exception:
                # fallback to params +/- 1.96*se
                ci_lower = float(coef - 1.96 * se)
                ci_upper = float(coef + 1.96 * se)
        else:
            ci_lower = float(coef - 1.96 * se)
            ci_upper = float(coef + 1.96 * se)

        # Multiplicative interpretation on raw rate scale
        try:
            exp_coef = float(np.exp(coef))
            exp_ci_lower = float(np.exp(ci_lower))
            exp_ci_upper = float(np.exp(ci_upper))
        except Exception:
            exp_coef = exp_ci_lower = exp_ci_upper = None

        # significance test (two-sided, alpha=0.05)
        significant = None
        if pv is not None:
            significant = bool(pv < 0.05)

        estimates[pred] = {
            'coef': float(coef),
            'se': float(se),
            'pvalue': float(pv) if pv is not None else None,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'exp_coef': exp_coef,
            'exp_ci_lower': exp_ci_lower,
            'exp_ci_upper': exp_ci_upper,
            'significant': significant,
            'note': "Dependent variable is log(nuts_opened + 0.5) - log(seconds). exp(coef) gives multiplicative change in nuts-per-second (pseudo-count adjusted)."
        }

    # Model-level diagnostics
    # number of observations
    nobs = None
    try:
        nobs = int(getattr(model_output, 'nobs', None) or getattr(model_output.model, 'endog', None).shape[0])
    except Exception:
        pass

    # number of groups
    n_groups = None
    try:
        groups_arr = getattr(model_output.model, 'groups', None)
        if groups_arr is not None:
            import numpy as _np
            n_groups = int(len(_np.unique(groups_arr)))
    except Exception:
        pass

    # random effect variance (for random intercept) and residual variance ('scale')
    random_effects_variance = None
    residual_variance = None
    try:
        cov_re = getattr(model_output, 'cov_re', None)
        if cov_re is not None:
            # if 1x1 matrix, take [0,0]
            random_effects_variance = float(cov_re.iloc[0, 0]) if hasattr(cov_re, 'iloc') else float(cov_re[0][0])
    except Exception:
        pass
    try:
        residual_variance = float(getattr(model_output, 'scale', None))
    except Exception:
        pass

    result_object = {
        'estimates': estimates,
        'nobs': nobs,
        'n_groups': n_groups,
        'random_effects_variance': random_effects_variance,
        'residual_variance': residual_variance
    }

    # Build a succinct textual summary automatically from the numeric results
    summary_lines = []
    for pred, vals in estimates.items():
        if vals.get('coef') is None:
            summary_lines.append(f"{pred}: not present in model output.")
            continue
        sig_mark = "significant" if vals['significant'] else "not significant"
        pv_str = f"p={vals['pvalue']:.3g}" if vals['pvalue'] is not None else "p=NA"
        summary_lines.append(
            f"{pred}: coef={vals['coef']:.3f}, SE={vals['se']:.3f}, {pv_str} ({sig_mark}); "
            f"95% CI [{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]; "
            f"exp(coef)={vals['exp_coef']:.3f} (95% CI [{vals['exp_ci_lower']:.3f}, {vals['exp_ci_upper']:.3f}])"
        )

    description = (
        "Extracted mixed-effects model estimates for predictors of nut-cracking efficiency.\n"
        "Interpretation: coefficients are on the log-rate scale (log(nuts_opened + 0.5) - log(seconds)). "
        "exp(coef) approximates the multiplicative change in nuts-per-second (pseudo-count adjusted) "
        "associated with a one-unit increase in the predictor (or for binary predictors, the change from 0 to 1).\n\n"
        "Model summary (predictors):\n" + "\n".join(summary_lines) + "\n\n"
        f"Observations: {nobs}, Groups: {n_groups}. Random-intercept variance (if available): {random_effects_variance}, residual variance (scale): {residual_variance}."
    )

    return {"object": result_object, "description": description}