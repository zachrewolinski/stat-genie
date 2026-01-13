def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of Reader View on reading speed for individuals with dyslexia
    from a statsmodels-like OLSResults (robust cov / clustered) object returned by the modeling function.

    Returns a dict with keys:
      - "object": dict containing the combined coefficient (ReaderView effect for dyslexic readers),
                  its standard error, t-stat, p-value, and 95% CI; also reports the separate
                  coefficients for RunningView and interaction term.
      - "description": textual interpretation (whether Reader View improves reading speed for
                       dyslexic individuals at alpha=0.05) and brief notes on what was computed.
    """
    import numpy as np
    from math import isfinite, isnan
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Obtain raw parameter values (array-like) and parameter names robustly
    params_raw = getattr(res, "params", None)

    # Try to extract parameter names from likely places
    param_names = None
    if params_raw is None:
        raise ValueError("model_output does not have attribute 'params'.")

    # If params is a pandas Series or similar with an index
    try:
        # Many statsmodels results have params as a pandas Series with .index
        param_index = getattr(params_raw, "index", None)
        if param_index is not None:
            param_names = [str(x) for x in param_index]
            params_values = np.asarray(params_raw)
        else:
            # params_raw might be a numpy array; try to get names from model or result attributes
            params_values = np.asarray(params_raw)
            # try common locations for names
            if hasattr(res, "model") and hasattr(res.model, "exog_names"):
                param_names = [str(x) for x in res.model.exog_names]
            elif hasattr(res, "param_names"):
                param_names = [str(x) for x in getattr(res, "param_names")]
            else:
                # fallback: create generic parameter names param_0, param_1, ...
                param_names = [f"param_{i}" for i in range(params_values.size)]
    except Exception as e:
        raise ValueError(f"Could not interpret 'params' from model_output: {e}")

    # Obtain covariance matrix (as array or DataFrame)
    cov_raw = None
    try:
        cov_raw = res.cov_params()
    except Exception:
        cov_raw = getattr(res, "cov_params_default", None)

    if cov_raw is None:
        raise ValueError("model_output does not have a covariance matrix via cov_params().")

    # Convert covariance to numpy array
    try:
        cov_mat = np.asarray(cov_raw)
    except Exception as e:
        raise ValueError(f"Could not convert covariance to array: {e}")

    # Validate dimensions
    if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if cov_mat.shape[0] != len(param_names):
        # try to handle case where cov is a DataFrame with labeled indices/columns
        try:
            # If cov_raw has .index and .columns, align by names
            idx = getattr(cov_raw, "index", None)
            cols = getattr(cov_raw, "columns", None)
            if idx is not None and cols is not None:
                cov_names = [str(x) for x in idx]
                if len(cov_names) == len(cols) and set(cov_names) == set([str(x) for x in cols]):
                    # reorder to param_names if possible
                    name_to_pos = {name: i for i, name in enumerate(cov_names)}
                    perm = [name_to_pos.get(n, None) for n in param_names]
                    if None not in perm:
                        cov_mat = np.asarray(cov_raw)[perm][:, perm]
                    else:
                        raise ValueError("Covariance DataFrame names do not align with parameter names.")
                else:
                    raise ValueError("Covariance DataFrame index/columns inconsistent.")
            else:
                raise ValueError("Covariance matrix shape does not match number of parameters.")
        except Exception as e:
            raise ValueError(f"Covariance matrix shape mismatch: {e}")

    # Map parameter names to indices
    name_to_idx = {name: idx for idx, name in enumerate(param_names)}

    # Helper to find parameter name for the interaction term (either order)
    def find_param(*tokens):
        """Find parameter name that contains all tokens in order or separated by ':'."""
        # exact join with colon first
        candidate1 = ":".join(tokens)
        if candidate1 in param_names:
            return candidate1
        # try reversed order
        candidate2 = ":".join(tokens[::-1])
        if candidate2 in param_names:
            return candidate2
        # fallback: find any name that contains both tokens (order-insensitive)
        for name in param_names:
            if all(tok in name for tok in tokens):
                return name
        return None

    running_name = find_param("RunningView")
    dys_name = find_param("DyslexiaIndicator")
    inter_name = find_param("RunningView", "DyslexiaIndicator")

    # Validate presence of required parameters
    missing = []
    if running_name is None or running_name not in name_to_idx:
        missing.append("RunningView")
    if inter_name is None or inter_name not in name_to_idx:
        missing.append("RunningView:DyslexiaIndicator (interaction)")
    if missing:
        raise ValueError(f"Required parameter(s) missing from model results: {missing}. "
                         f"Available parameter names: {param_names}")

    # Helper to get parameter value by name
    def get_param_value(name):
        idx = name_to_idx.get(name)
        if idx is None or idx >= params_values.size:
            raise KeyError(f"Parameter '{name}' not found in params.")
        return float(params_values[idx])

    # Extract coefficients
    beta_run = get_param_value(running_name)
    beta_int = get_param_value(inter_name)
    # Combined effect for dyslexic readers = RunningView + interaction
    beta_comb = beta_run + beta_int

    # Extract variances and covariance using indices
    try:
        i_run = name_to_idx[running_name]
        i_int = name_to_idx[inter_name]
        var_run = float(cov_mat[i_run, i_run])
        var_int = float(cov_mat[i_int, i_int])
        cov_run_int = float(cov_mat[i_run, i_int])
    except Exception as e:
        raise ValueError(f"Could not extract variance/covariance entries for parameters: {e}")

    # Compute standard error for the linear combination: Var(a+b) = Var(a)+Var(b)+2Cov(a,b)
    var_comb = var_run + var_int + 2.0 * cov_run_int
    se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 and isfinite(var_comb) else float("nan")

    # t-stat and p-value
    if se_comb == 0 or not isfinite(se_comb):
        t_comb = float("nan")
        p_comb = float("nan")
        t_crit = 1.96
    else:
        t_comb = beta_comb / se_comb
        df_resid = getattr(res, "df_resid", None)
        # df_resid might be an array-like or numpy scalar; coerce to float if possible
        try:
            df_val = float(df_resid) if df_resid is not None else None
        except Exception:
            df_val = None
        if stats is not None and df_val is not None and isfinite(df_val) and df_val > 0:
            p_comb = 2.0 * stats.t.sf(abs(t_comb), df_val)
            t_crit = stats.t.ppf(0.975, df_val)
        elif stats is not None:
            p_comb = 2.0 * stats.norm.sf(abs(t_comb))
            t_crit = stats.norm.ppf(0.975)
        else:
            p_comb = float("nan")
            t_crit = 1.96

    ci_lower = beta_comb - t_crit * se_comb if isfinite(se_comb) else float("nan")
    ci_upper = beta_comb + t_crit * se_comb if isfinite(se_comb) else float("nan")

    # Also report the main RunningView coefficient and interaction with their robust p-values and CIs
    def param_summary(name):
        b = get_param_value(name)
        idx = name_to_idx[name]
        se = float(np.sqrt(float(cov_mat[idx, idx]))) if cov_mat.shape[0] > idx else float("nan")
        if se == 0 or not isfinite(se):
            tval = float("nan")
            pval = float("nan")
            crit = 1.96
        else:
            tval = b / se
            df_resid = getattr(res, "df_resid", None)
            try:
                df_val = float(df_resid) if df_resid is not None else None
            except Exception:
                df_val = None
            if stats is not None and df_val is not None and isfinite(df_val) and df_val > 0:
                pval = 2.0 * stats.t.sf(abs(tval), df_val)
                crit = stats.t.ppf(0.975, df_val)
            elif stats is not None:
                pval = 2.0 * stats.norm.sf(abs(tval))
                crit = stats.norm.ppf(0.975)
            else:
                pval = float("nan")
                crit = 1.96
        ci_l = b - crit * se if isfinite(se) else float("nan")
        ci_u = b + crit * se if isfinite(se) else float("nan")
        return {"coef": b, "se": se, "t": tval, "p_value": pval, "ci_lower": ci_l, "ci_upper": ci_u}

    run_summary = param_summary(running_name)
    int_summary = param_summary(inter_name)

    # Decide interpretation for dyslexic readers
    alpha = 0.05
    if not isnan(p_comb):
        if (beta_comb > 0) and (p_comb < alpha):
            interpretation = ("There is statistically significant evidence (two-sided p = "
                              f"{p_comb:.3g}) that enabling Reader View increases reading speed "
                              f"for readers with dyslexia by {beta_comb:.3g} wps on average "
                              f"(95% CI [{ci_lower:.3g}, {ci_upper:.3g}]).")
        elif (beta_comb < 0) and (p_comb < alpha):
            interpretation = ("There is statistically significant evidence (two-sided p = "
                              f"{p_comb:.3g}) that enabling Reader View decreases reading speed "
                              f"for readers with dyslexia by {abs(beta_comb):.3g} wps on average "
                              f"(95% CI [{ci_lower:.3g}, {ci_upper:.3g}]).")
        else:
            interpretation = ("No statistically significant effect of Reader View on reading speed "
                              f"for readers with dyslexia was detected (estimated effect = {beta_comb:.3g} wps, "
                              f"95% CI [{ci_lower:.3g}, {ci_upper:.3g}], two-sided p = {p_comb:.3g}).")
    else:
        interpretation = ("Could not compute a valid test for the combined effect (NaN encountered). "
                          "Check model_output and its covariance matrix.")

    result_object = {
        "combined_effect_for_dyslexic_readers": {
            "coef": beta_comb,
            "se": se_comb,
            "t": t_comb,
            "p_value": p_comb,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "units": "words per second (wps)",
            "note": "This is RunningView + RunningView:DyslexiaIndicator, i.e. the effect of Reader View when DyslexiaIndicator=1"
        },
        "running_view_main": run_summary,
        "interaction_running_x_dyslexia": int_summary,
        "model_param_names": param_names,
        "alpha": alpha
    }

    description = (
        interpretation
        + " The returned 'object' contains the numeric estimates (coef, se, t, p, 95% CI) for the combined effect "
        "and also the separate RunningView and interaction parameter summaries. "
        "Decision rule used: two-sided test at alpha=0.05; positive coef = increase in reading speed (wps)."
    )

    return {"object": result_object, "description": description}