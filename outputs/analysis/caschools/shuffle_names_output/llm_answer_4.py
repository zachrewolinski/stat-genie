def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, and 95% CI for the student-teacher ratio
    (and its log specification) from the provided model_output dict and returns an interpreted
    summary.

    Returned dict has keys:
      - "object": a dict with numeric results for linear and log specifications (or None when unavailable)
      - "description": a short interpretation answering whether a lower STR is associated with higher performance
    """
    import math

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def summarize_result(res, var_name):
        """
        Given a fitted statsmodels-like result object and the name of a variable,
        return a dict with coef, se, pvalue, ci95 = [low, high], and raw presence flag.
        Returns None if the result object or variable is not available.
        """
        if res is None:
            return None

        # Try to get params (pandas Series is typical)
        params = getattr(res, 'params', None)
        if params is None:
            return None

        # Determine index/position of variable
        var_index = None
        try:
            # If params has index (pandas Series), check membership
            if hasattr(params, 'index') and var_name in params.index:
                var_index = list(params.index).index(var_name)
            else:
                # try model.exog_names (statsmodels)
                exog_names = None
                if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                    exog_names = list(res.model.exog_names)
                elif hasattr(res, 'params') and hasattr(res.params, 'index'):
                    exog_names = list(res.params.index)
                if exog_names and var_name in exog_names:
                    var_index = exog_names.index(var_name)
                else:
                    # last resort: try to find element by string match in params index
                    try:
                        var_index = [str(x) for x in params.index].index(var_name)
                    except Exception:
                        var_index = None
        except Exception:
            var_index = None

        if var_index is None:
            return None

        # Extract coef, se, pvalue
        try:
            coef = safe_float(params[var_name]) if hasattr(params, 'index') and var_name in params.index else safe_float(list(params)[var_index])
        except Exception:
            coef = None

        bse = getattr(res, 'bse', None)
        se = None
        if bse is not None:
            try:
                se = safe_float(bse[var_name]) if hasattr(bse, 'index') and var_name in bse.index else safe_float(list(bse)[var_index])
            except Exception:
                se = None

        pvalues = getattr(res, 'pvalues', None)
        pvalue = None
        if pvalues is not None:
            try:
                pvalue = safe_float(pvalues[var_name]) if hasattr(pvalues, 'index') and var_name in pvalues.index else safe_float(list(pvalues)[var_index])
            except Exception:
                pvalue = None

        # Confidence interval
        ci_low = ci_high = None
        conf = None
        try:
            # conf_int might be a method
            conf = res.conf_int() if callable(getattr(res, 'conf_int', None)) else getattr(res, 'conf_int', None)
        except Exception:
            conf = None

        if conf is not None:
            try:
                # If conf is a DataFrame or array with index
                if hasattr(conf, 'loc') and var_name in getattr(conf, 'index', []):
                    row = conf.loc[var_name]
                    ci_low = safe_float(row[0])
                    ci_high = safe_float(row[1])
                else:
                    # treat as ndarray-like
                    ci_low = safe_float(conf[var_index][0])
                    ci_high = safe_float(conf[var_index][1])
            except Exception:
                ci_low = ci_high = None

        # nobs if available
        nobs = None
        try:
            nobs = int(getattr(res, 'nobs')) if getattr(res, 'nobs', None) is not None else None
        except Exception:
            try:
                nobs = int(getattr(res, 'df_resid') + getattr(res, 'df_model') + 1)
            except Exception:
                nobs = None

        return {
            'var': var_name,
            'coef': coef,
            'se': se,
            'pvalue': pvalue,
            'ci95': [ci_low, ci_high],
            'nobs': nobs
        }

    # Begin extraction
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict. Cannot extract results."
        }

    n_obs = model_output.get('n_obs', None)
    ols = model_output.get('ols_model', None)
    ols_log = model_output.get('ols_model_log', None)

    # If there are zero observations or models are not present, return a clear message
    if (n_obs is None and ols is None and ols_log is None) or (isinstance(n_obs, (int, float)) and n_obs == 0):
        return {
            "object": None,
            "description": "No analytic sample available (n_obs is 0 or models are missing). Cannot determine whether lower student-teacher ratio is associated with higher academic performance."
        }

    # Summarize linear specification (StudentTeacherRatio)
    linear_summary = summarize_result(ols, 'StudentTeacherRatio')

    # Summarize log specification (log_STR)
    log_summary = summarize_result(ols_log, 'log_STR')

    # Build interpretation text
    interpretations = []
    overall_answer = None  # will be 'yes', 'no', 'inconclusive', or None

    def interpret_one(summ, label):
        if summ is None:
            return None
        coef = summ.get('coef')
        p = summ.get('pvalue')
        ci = summ.get('ci95')
        # Decide significance at 0.05 if p-value present
        sig = None
        if p is not None:
            try:
                sig = (p < 0.05)
            except Exception:
                sig = None
        # Interpret sign
        sign = None
        if coef is not None:
            if coef < 0:
                sign = 'negative'
            elif coef > 0:
                sign = 'positive'
            else:
                sign = 'zero'

        interp = {
            'model': label,
            'coef': coef,
            'pvalue': p,
            'ci95': ci,
            'sign': sign,
            'significant_at_0.05': sig
        }

        # Generate a short textual interpretation
        if coef is None:
            text = f"{label}: result available but coefficient could not be read."
        else:
            # Relationship mapping: negative coef => lower STR associated with higher performance
            if sign == 'negative' and sig is True:
                text = (f"{label}: Coefficient = {coef:.4g} (p = {p:.3g}). Statistically significant and negative: "
                        "this indicates that higher student-teacher ratios (more students per teacher) are associated "
                        "with lower academic performance — equivalently, a lower student-teacher ratio is associated with higher performance.")
            elif sign == 'negative' and (sig is False or sig is None):
                text = (f"{label}: Coefficient = {coef:.4g} (p = {p if p is not None else 'NA'}). Negative but not statistically significant: "
                        "point estimate suggests lower STR may be associated with higher performance, but evidence is weak/inconclusive.")
            elif sign == 'positive' and sig is True:
                text = (f"{label}: Coefficient = {coef:.4g} (p = {p:.3g}). Statistically significant and positive: "
                        "this indicates that higher student-teacher ratios are associated with higher academic performance — "
                        "so lower STR would be associated with lower performance (contrary to expectation).")
            elif sign == 'positive' and (sig is False or sig is None):
                text = (f"{label}: Coefficient = {coef:.4g} (p = {p if p is not None else 'NA'}). Positive but not statistically significant: "
                        "point estimate suggests higher STR might be associated with higher performance, but evidence is weak/inconclusive.")
            else:
                text = f"{label}: Coefficient = {coef:.4g} (p = {p if p is not None else 'NA'}). No clear directional effect."
        return interp, text

    lin_res = interpret_one(linear_summary, "Linear (level STR)")
    log_res = interpret_one(log_summary, "Log specification (log STR)")

    if lin_res is not None:
        interpretations.append(lin_res[1])
    if log_res is not None:
        interpretations.append(log_res[1])

    # Combine overall assessment if possible
    # If linear is significant negative -> answer yes. If significant positive -> no.
    # If neither significant -> inconclusive.
    decision = None
    if linear_summary and linear_summary.get('coef') is not None and linear_summary.get('pvalue') is not None:
        if linear_summary['pvalue'] < 0.05:
            decision = 'yes' if linear_summary['coef'] < 0 else 'no'
    if decision is None and log_summary and log_summary.get('coef') is not None and log_summary.get('pvalue') is not None:
        if log_summary['pvalue'] < 0.05:
            # For log coef: negative -> yes
            decision = 'yes' if log_summary['coef'] < 0 else 'no'

    if decision is None:
        overall_text = "Overall: Evidence is inconclusive based on the available model output (no statistically significant, consistent estimate found or models missing)."
    elif decision == 'yes':
        overall_text = "Overall answer: Yes — the estimated association indicates that a lower student-teacher ratio is associated with higher academic performance (statistically significant)."
    else:
        overall_text = "Overall answer: No — the estimated association indicates that a lower student-teacher ratio is associated with lower academic performance (statistically significant in the opposite direction)."

    description_lines = []
    if isinstance(n_obs, int):
        description_lines.append(f"n_obs = {n_obs}")
    description_lines.extend(interpretations)
    description_lines.append(overall_text)
    description = " ".join(description_lines) if description_lines else overall_text

    result_object = {
        'n_obs': n_obs,
        'linear_summary': linear_summary,
        'log_summary': log_summary
    }

    return {
        "object": result_object,
        "description": description
    }