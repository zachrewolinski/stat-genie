def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of IsHuman from the fitted model output.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, z, p, 95% CI on log-odds,
                  odds ratio and its 95% CI, boolean 'significant', and a short conclusion)
      - "description": brief plain-English explanation of what the numbers mean.

    Accepts either the RobustResultsWrapper used in the modeling code (has .params,
    .bse, .zvalues, .pvalues, .cov_params) or a statsmodels results object.
    """
    import numpy as np
    from scipy import stats

    # Helper to safely format numbers
    def _fmt(x, ndigits=4, none_str="NA"):
        if x is None:
            return none_str
        try:
            return f"{x:.{ndigits}f}"
        except Exception:
            return str(x)

    # Helper to safely get parameter names and values
    params = None
    bse = None
    z = None
    p = None
    cov = None

    # Try to extract commonly available attributes
    try:
        params = model_output.params
    except Exception:
        params = getattr(model_output, 'params', None)

    try:
        bse = model_output.bse
    except Exception:
        bse = getattr(model_output, 'bse', None)

    try:
        z = model_output.zvalues
    except Exception:
        z = getattr(model_output, 'tvalues', None) or getattr(model_output, 'zvalues', None)

    try:
        p = model_output.pvalues
    except Exception:
        p = getattr(model_output, 'pvalues', None)

    # Covariance matrix (robust cov if provided)
    cov = getattr(model_output, 'cov_params', None)
    # If cov is a method, call it
    if callable(cov):
        try:
            cov = cov()
        except Exception:
            cov = None

    # Ensure params can be indexed by name; handle pandas Series or dict-like
    param_names = None
    if params is None:
        raise ValueError("model_output has no 'params' attribute accessible.")
    # If params is a pandas Series, get index; otherwise try keys
    try:
        param_names = list(params.index)
    except Exception:
        try:
            param_names = list(params.keys())
        except Exception:
            # fallback to treating params as array and try to get names from model object
            param_names = None

    # Determine where 'IsHuman' is located
    target_name = 'IsHuman'
    loc = None
    coef = None
    se = None
    zval = None
    pval = None

    if param_names is not None and target_name in param_names:
        loc = param_names.index(target_name)
        # extract numeric values
        try:
            coef = float(params[target_name])
        except Exception:
            try:
                coef = float(params.iloc[loc]) if hasattr(params, 'iloc') else float(params[loc])
            except Exception:
                coef = None
        # standard error: preferentially from bse, else from cov matrix diag
        if bse is not None:
            try:
                se = float(bse[target_name])
            except Exception:
                try:
                    se = float(bse.iloc[loc])
                except Exception:
                    se = None
        else:
            se = None
        if se is None and cov is not None:
            # try to take sqrt of diag element
            cov_arr = np.asarray(cov)
            try:
                se = float(np.sqrt(np.abs(cov_arr[loc, loc])))
            except Exception:
                se = None
        # z and p
        if z is not None:
            try:
                zval = float(z[target_name])
            except Exception:
                try:
                    zval = float(z.iloc[loc])
                except Exception:
                    zval = None
        else:
            zval = None
        if p is not None:
            try:
                pval = float(p[target_name])
            except Exception:
                try:
                    pval = float(p.iloc[loc])
                except Exception:
                    pval = None
        else:
            pval = None
    else:
        # If param names not available, try positional extraction
        # Try to get params as numpy array
        try:
            params_arr = np.asarray(params)
        except Exception:
            raise ValueError("Could not determine parameter names or extract 'IsHuman'.")
        # Try to find 'IsHuman' in model's exog names if available
        loc = None
        try:
            exog_names = model_output._orig.model.exog_names
            if target_name in exog_names:
                loc = list(exog_names).index(target_name)
        except Exception:
            pass
        if loc is None:
            # As a last resort, raise error
            raise ValueError("Could not locate parameter 'IsHuman' in model output.")
        try:
            coef = float(params_arr[loc])
        except Exception:
            coef = None
        se = None
        if bse is not None:
            try:
                se = float(np.asarray(bse)[loc])
            except Exception:
                se = None
        elif cov is not None:
            try:
                cov_arr = np.asarray(cov)
                se = float(np.sqrt(np.abs(cov_arr[loc, loc])))
            except Exception:
                se = None
        try:
            zval = float(np.asarray(z)[loc]) if z is not None else None
        except Exception:
            zval = None
        try:
            pval = float(np.asarray(p)[loc]) if p is not None else None
        except Exception:
            pval = None

    # If any of se, z, p are missing, compute from available pieces
    if se is None and cov is not None and loc is not None:
        try:
            cov_arr = np.asarray(cov)
            se = float(np.sqrt(np.abs(cov_arr[loc, loc])))
        except Exception:
            se = None
    if zval is None and se is not None and coef is not None:
        try:
            zval = coef / se
        except Exception:
            zval = None
    if pval is None and zval is not None:
        try:
            pval = 2 * (1 - stats.norm.cdf(abs(zval)))
        except Exception:
            pval = None

    # Compute 95% CI on log-odds and odds ratio
    z_crit = stats.norm.ppf(0.975)
    if se is not None and coef is not None:
        ci_low = coef - z_crit * se
        ci_high = coef + z_crit * se
    else:
        ci_low = ci_high = None

    # Odds ratio and its CI
    try:
        odds_ratio = float(np.exp(coef)) if coef is not None else None
    except Exception:
        odds_ratio = None
    if ci_low is not None and ci_high is not None:
        try:
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        except Exception:
            or_ci = (None, None)
    else:
        or_ci = (None, None)

    significant = (pval is not None) and (pval < 0.05)

    # Build concise conclusion text
    if significant:
        if coef is not None:
            direction = 'positive' if coef > 0 else ('negative' if coef < 0 else 'zero')
        else:
            direction = 'directional'
        concl = (f"The IsHuman coefficient is {direction} and statistically significant "
                 f"(coef = {_fmt(coef,4)}, SE = {_fmt(se,4)}, z = {_fmt(zval,3)}, p = {_fmt(pval,4)}). "
                 "After controlling for age, prob_male, and tooth_class, modern humans "
                 "have higher AMTL compared with the reference group.")
    else:
        concl = (f"The IsHuman coefficient is not statistically significant "
                 f"(coef = {_fmt(coef,4)}, SE = {_fmt(se,4)}, z = {_fmt(zval,3)}, p = {_fmt(pval,4)}). "
                 "No evidence that modern humans differ from non-human primates in AMTL "
                 "after controls.")

    # Full interpretation with odds ratio
    if (odds_ratio is not None) and (or_ci[0] is not None):
        concl += (f" The estimated odds ratio is {_fmt(odds_ratio,2)} "
                  f"(95% CI: {_fmt(or_ci[0],2)}–{_fmt(or_ci[1],2)}), meaning modern humans have about "
                  f"{_fmt(odds_ratio,2)} times the odds of AMTL compared with non-human primates "
                  "holding other variables constant.")

    result_object = {
        'parameter': 'IsHuman',
        'coef_log_odds': coef,
        'std_error': se,
        'z_value': zval,
        'p_value': pval,
        'ci_log_odds_95': (ci_low, ci_high),
        'odds_ratio': odds_ratio,
        'odds_ratio_95_ci': or_ci,
        'significant': bool(significant),
        'conclusion': concl
    }

    description = (
        "Extracted effect of IsHuman from the fitted binomial GLM (logit link). "
        "A positive coefficient indicates higher log-odds (and thus higher probability) "
        "of antemortem tooth loss in Homo sapiens relative to the non-human primate "
        "reference, after controlling for age, prob_male, and tooth_class. "
        "The returned 'object' contains the coefficient, robust SE (if available), z and p "
        "values, 95% CI on the log-odds scale and on the odds-ratio scale, and a short "
        "conclusion about statistical significance."
    )

    return {'object': result_object, 'description': description}