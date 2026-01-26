def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-stat, p-value, and 95% CI for the
    StudentTeacherRatio coefficient from a statsmodels-like results object (regular
    or robust results wrapper), or from dict-like/sequence inputs. Returns a dict
    with numeric results under "object" and a short interpretation under "description".
    """
    import numpy as np
    from math import isnan, sqrt
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    def safe_get_mapping_value(mapping, key, idx=None):
        """Try to extract value by key from various mapping/sequence types."""
        try:
            if mapping is None:
                return None
            # dict-like
            if isinstance(mapping, dict):
                if key in mapping:
                    return mapping[key]
                # maybe mapping['params'] exists and is a dict/sequence
                if 'params' in mapping and isinstance(mapping['params'], (dict, list, tuple, np.ndarray)):
                    m = mapping['params']
                    if isinstance(m, dict) and key in m:
                        return m[key]
                    if idx is not None and isinstance(m, (list, tuple, np.ndarray)) and idx < len(m):
                        return m[idx]
                return None
            # pandas Series / DataFrame-like
            if hasattr(mapping, 'loc') and hasattr(mapping, 'index'):
                # Series: mapping.loc[key]
                try:
                    return mapping.loc[key]
                except Exception:
                    # maybe key is not present
                    pass
            # mapping with keys()
            if hasattr(mapping, 'keys'):
                try:
                    if key in mapping.keys():
                        return mapping[key]
                except Exception:
                    pass
            # sequence-like
            if isinstance(mapping, (list, tuple, np.ndarray)):
                if idx is not None and 0 <= idx < len(mapping):
                    return mapping[idx]
            # fallback: getattr
            if hasattr(mapping, key):
                return getattr(mapping, key)
        except Exception:
            return None
        return None

    # Build parameter name list robustly from many possible places
    param_index = None
    try:
        # If model_output itself is a dict mapping names to values
        if isinstance(res, dict):
            # prefer res['params'] if present
            if 'params' in res and isinstance(res['params'], dict):
                param_index = list(res['params'].keys())
            else:
                # treat top-level keys as parameters if they look like parameter names
                param_index = list(res.keys())
        else:
            # If res has params attribute, inspect it
            if hasattr(res, 'params'):
                p = res.params
                if hasattr(p, 'index'):
                    # pandas Series
                    try:
                        param_index = list(p.index)
                    except Exception:
                        param_index = None
                elif hasattr(p, 'keys'):
                    try:
                        param_index = list(p.keys())
                    except Exception:
                        param_index = None
                elif isinstance(p, (list, tuple, np.ndarray)):
                    # try to get names from model metadata
                    if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                        try:
                            param_index = list(res.model.exog_names)
                        except Exception:
                            param_index = [f'param_{i}' for i in range(len(p))]
                    elif hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'param_names'):
                        try:
                            param_index = list(res.model.data.param_names)
                        except Exception:
                            param_index = [f'param_{i}' for i in range(len(p))]
                    else:
                        param_index = [f'param_{i}' for i in range(len(p))]
                else:
                    # last attempt: try to coerce p to list of names
                    try:
                        param_index = list(p)
                    except Exception:
                        param_index = None
            # If still not found, try model.exog_names
            if not param_index and hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                try:
                    param_index = list(res.model.exog_names)
                except Exception:
                    param_index = None
            # try data param names
            if not param_index and hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'param_names'):
                try:
                    param_index = list(res.model.data.param_names)
                except Exception:
                    param_index = None
            # try attribute 'names'
            if not param_index and hasattr(res, 'names'):
                try:
                    param_index = list(res.names)
                except Exception:
                    param_index = None
    except Exception:
        param_index = None

    if not param_index:
        # As a last resort, if res.params is a sequence infer synthetic names
        try:
            if hasattr(res, 'params') and isinstance(res.params, (list, tuple, np.ndarray)):
                param_index = [f'param_{i}' for i in range(len(res.params))]
            elif isinstance(res, (list, tuple, np.ndarray)):
                param_index = [f'param_{i}' for i in range(len(res))]
        except Exception:
            param_index = None

    if not param_index:
        available = []
        try:
            if isinstance(res, dict):
                available = list(res.keys())
            elif hasattr(res, '__dict__'):
                available = list(res.__dict__.keys())
        except Exception:
            available = []
        raise KeyError(
            "Could not determine parameter names from the model_output object. "
            "Available attributes/keys seen: {}".format(available)
        )

    # Find the parameter name corresponding to StudentTeacherRatio (allow some fuzziness)
    candidates = [n for n in param_index if isinstance(n, str) and 'StudentTeacherRatio' in n]
    if not candidates:
        candidates = [n for n in param_index if isinstance(n, str) and ('student' in n.lower() and 'teacher' in n.lower())]
    if not candidates:
        raise KeyError("Could not find a parameter matching 'StudentTeacherRatio' in model params. "
                       "Available params: {}".format(param_index))
    param_name = candidates[0]

    # Helper to find index of param_name in param_index (if possible)
    try:
        param_idx = param_index.index(param_name)
    except Exception:
        param_idx = None

    # Extract coefficient
    coef = None
    # Try multiple places for coefficient: res.params, res (if dict), res.params array by index, res.__dict__ etc.
    try:
        if hasattr(res, 'params'):
            val = safe_get_mapping_value(res.params, param_name, param_idx)
            if val is None and param_idx is not None and isinstance(res.params, (list, tuple, np.ndarray)):
                val = res.params[param_idx]
            if val is not None:
                coef = float(val)
    except Exception:
        coef = None
    if coef is None:
        # try top-level dict-like
        try:
            val = safe_get_mapping_value(res, param_name, param_idx)
            if val is not None:
                coef = float(val)
        except Exception:
            coef = None
    if coef is None:
        # try attribute named like param_name
        try:
            if hasattr(res, param_name):
                coef = float(getattr(res, param_name))
        except Exception:
            coef = None
    if coef is None:
        raise KeyError(f"Could not extract coefficient for parameter '{param_name}' from model_output.")

    # Extract standard error (try common locations)
    se = None
    try:
        if hasattr(res, 'bse'):
            val = safe_get_mapping_value(res.bse, param_name, param_idx)
            if val is None and param_idx is not None and isinstance(res.bse, (list, tuple, np.ndarray)):
                val = res.bse[param_idx]
            if val is not None:
                se = float(val)
    except Exception:
        se = None

    if se is None:
        try:
            # Try variance from cov_params
            if hasattr(res, 'cov_params'):
                cov = res.cov_params()
                # cov may be DataFrame or ndarray
                if hasattr(cov, 'loc'):
                    # DataFrame
                    try:
                        var = cov.loc[param_name, param_name]
                        se = float(np.sqrt(float(var)))
                    except Exception:
                        # If DataFrame with numeric index
                        if param_idx is not None:
                            try:
                                var = cov.iloc[param_idx, param_idx]
                                se = float(np.sqrt(float(var)))
                            except Exception:
                                se = None
                else:
                    # ndarray
                    try:
                        idx = param_idx if param_idx is not None else 0
                        var = cov[idx, idx]
                        se = float(np.sqrt(float(var)))
                    except Exception:
                        se = None
        except Exception:
            se = None

    # Compute t-statistic if not present
    tstat = None
    try:
        if hasattr(res, 'tvalues'):
            val = safe_get_mapping_value(res.tvalues, param_name, param_idx)
            if val is None and param_idx is not None and isinstance(res.tvalues, (list, tuple, np.ndarray)):
                val = res.tvalues[param_idx]
            if val is not None:
                tstat = float(val)
    except Exception:
        tstat = None
    if tstat is None and se is not None and se != 0:
        try:
            tstat = float(coef / se)
        except Exception:
            tstat = None

    # Extract p-value (try direct attribute, otherwise compute via t-dist if possible)
    pvalue = None
    try:
        if hasattr(res, 'pvalues'):
            val = safe_get_mapping_value(res.pvalues, param_name, param_idx)
            if val is None and param_idx is not None and isinstance(res.pvalues, (list, tuple, np.ndarray)):
                val = res.pvalues[param_idx]
            if val is not None:
                pvalue = float(val)
    except Exception:
        pvalue = None

    if pvalue is None:
        # fallback: compute two-sided p-value from t and df_resid if scipy is available
        try:
            df = None
            if hasattr(res, 'df_resid'):
                df = float(res.df_resid)
            elif hasattr(res, 'df_resid'):
                df = float(getattr(res, 'df_resid'))
            if tstat is not None and df is not None and stats is not None:
                pvalue = float(2.0 * (1.0 - stats.t.cdf(abs(tstat), df)))
        except Exception:
            pvalue = None

    # Confidence interval: try res.conf_int(), else compute from tcrit if possible
    ci_lower = ci_upper = None
    try:
        if hasattr(res, 'conf_int'):
            ci = res.conf_int(alpha=0.05)
            if hasattr(ci, 'loc'):
                # pandas DataFrame-like
                try:
                    ci_lower = float(ci.loc[param_name, 0])
                    ci_upper = float(ci.loc[param_name, 1])
                except Exception:
                    # maybe integer index
                    if param_idx is not None:
                        ci_lower = float(ci.iloc[param_idx, 0])
                        ci_upper = float(ci.iloc[param_idx, 1])
            else:
                # numpy array-like
                idx = param_idx if param_idx is not None else 0
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    if (ci_lower is None or ci_upper is None) and se is not None:
        try:
            df = None
            if hasattr(res, 'df_resid'):
                df = float(res.df_resid)
            if stats is not None and df is not None:
                tcrit = stats.t.ppf(1 - 0.025, df)
                ci_lower = float(coef - tcrit * se)
                ci_upper = float(coef + tcrit * se)
            elif se is not None:
                # fallback to normal approx
                z = 1.96
                ci_lower = float(coef - z * se)
                ci_upper = float(coef + z * se)
        except Exception:
            ci_lower = ci_upper = None

    # Decision and interpretation
    significance = None
    if pvalue is not None:
        significance = (pvalue < 0.05)

    # Direction & implication text
    if coef < 0:
        direction = "negative"
        implication = ("A negative coefficient means that higher student-teacher ratios (more students per teacher) "
                       "are associated with lower AvgScore; therefore LOWER student-teacher ratios (fewer students per teacher) "
                       "are associated with HIGHER academic performance.")
    elif coef > 0:
        direction = "positive"
        implication = ("A positive coefficient means that higher student-teacher ratios (more students per teacher) "
                       "are associated with HIGHER AvgScore; therefore LOWER student-teacher ratios (fewer students per teacher) "
                       "would be associated with LOWER academic performance.")
    else:
        direction = "zero"
        implication = "Coefficient is (approximately) zero; no association detected."

    # Build the object to return
    numeric_result = {
        "parameter_name": param_name,
        "coef": float(coef),
        "std_error": (None if se is None or (isinstance(se, float) and isnan(se)) else float(se)),
        "t_stat": (None if tstat is None or (isinstance(tstat, float) and isnan(tstat)) else float(tstat)),
        "p_value": (None if pvalue is None or (isinstance(pvalue, float) and isnan(pvalue)) else float(pvalue)),
        "ci_95_lower": (None if ci_lower is None or (isinstance(ci_lower, float) and isnan(ci_lower)) else float(ci_lower)),
        "ci_95_upper": (None if ci_upper is None or (isinstance(ci_upper, float) and isnan(ci_upper)) else float(ci_upper)),
        "significant_at_0_05": significance,
        "direction": direction
    }

    # Short description interpreting the results in context
    if significance is True:
        sig_text = "Statistically significant at the 0.05 level."
    elif significance is False:
        sig_text = "Not statistically significant at the 0.05 level."
    else:
        sig_text = "Statistical significance could not be determined."

    # Safe formatted strings
    p_text = f"{numeric_result['p_value']:.4g}" if numeric_result['p_value'] is not None else "NA"
    se_text = f"{numeric_result['std_error']:.4g}" if numeric_result['std_error'] is not None else "NA"
    ci_lower_text = f"{numeric_result['ci_95_lower']:.4g}" if numeric_result['ci_95_lower'] is not None else "NA"
    ci_upper_text = f"{numeric_result['ci_95_upper']:.4g}" if numeric_result['ci_95_upper'] is not None else "NA"

    description = (
        f"Estimate for '{param_name}': coefficient = {numeric_result['coef']:.4g}, "
        f"SE = {se_text}, p = {p_text}. "
        f"95% CI = [{ci_lower_text}, {ci_upper_text}]. "
        f"{sig_text} {implication} "
        "Interpretation: the coefficient gives the change in district average academic performance (AvgScore) "
        "associated with a one-unit increase in StudentTeacherRatio (one more student per teacher)."
    )

    return {"object": numeric_result, "description": description}