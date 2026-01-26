def extract_final_answer(model_output):
    """
    Extracts coefficients, SEs, p-values, 95% CIs, and interpretable effect sizes for the fixed effects
    from a statsmodels MixedLMResults (or MixedLMResultsWrapper) object.

    Returns:
      dict with keys:
        - "object": a dictionary with numeric results for each fixed-effect term including:
            estimate, std_error, t_or_z, p_value, 95% CI, exponentiated param and CI (multiplicative change in rate)
        - "description": human-readable explanation of what the numbers mean for age, sex_male, and help_received.
    """
    import numpy as np

    # Helper: safe attribute getter that calls method if callable
    def _safe_get_attr(obj, name):
        if hasattr(obj, name):
            attr = getattr(obj, name)
            return attr() if callable(attr) else attr
        return None

    # Helper to retrieve a scalar from container by name (if available) or by index
    def _get_value(container, name, idx):
        if container is None:
            return None
        # Pandas-like Series/DataFrame
        try:
            import pandas as _pd
            if isinstance(container, (_pd.Series, _pd.DataFrame)):
                if name in container.index:
                    val = container.loc[name]
                    # If DataFrame row, return first element; if Series scalar, ensure scalar
                    if isinstance(val, _pd.Series):
                        # prefer first element of the row/Series
                        return val.iloc[0]
                    return val
        except Exception:
            pass
        # Try dictionary-like access
        try:
            if hasattr(container, "get"):
                # container.get may return None if key not present
                val = container.get(name)
                if val is not None:
                    return val
        except Exception:
            pass
        # Fallback to __getitem__ by index or name
        try:
            return container[name]
        except Exception:
            try:
                return container[idx]
            except Exception:
                # Last-resort: try to convert to ndarray and index
                try:
                    arr = np.asarray(container)
                    return arr[idx]
                except Exception:
                    raise

    # Extract basic arrays/Series
    params = _safe_get_attr(model_output, "params")
    bse = _safe_get_attr(model_output, "bse")
    # tvalues may be called tvalues or zvalues depending on model/results
    tvals = _safe_get_attr(model_output, "tvalues")
    if tvals is None:
        tvals = _safe_get_attr(model_output, "zvalues")
    # pvalues may or may not be present; compute if missing
    pvals = _safe_get_attr(model_output, "pvalues")
    conf_int = None
    try:
        conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    # Ensure params and bse exist
    if params is None or bse is None:
        raise ValueError("Model output does not contain params or bse attributes required for extraction.")

    # Try to detect pandas and keep things as pandas structures when possible
    pd = None
    try:
        import pandas as pd
    except Exception:
        pd = None
    else:
        pd = pd

    # If params/bse are not pandas Series but pandas is available, convert to Series
    if pd is not None:
        try:
            if not isinstance(params, pd.Series):
                # If params has an index attribute (e.g., returned as ndarray with index elsewhere), prefer that index
                # Otherwise create a simple RangeIndex
                try:
                    index = params.index if hasattr(params, "index") else None
                except Exception:
                    index = None
                params = pd.Series(np.asarray(params), index=index)
            if not isinstance(bse, pd.Series):
                try:
                    index = bse.index if hasattr(bse, "index") else (params.index if hasattr(params, "index") else None)
                except Exception:
                    index = None
                bse = pd.Series(np.asarray(bse), index=index)
            if conf_int is not None and not isinstance(conf_int, pd.DataFrame):
                # try to coerce conf_int into DataFrame matching params index
                try:
                    conf_int = pd.DataFrame(conf_int, index=params.index, columns=[0, 1])
                except Exception:
                    try:
                        conf_int = pd.DataFrame(conf_int)
                    except Exception:
                        conf_int = conf_int
        except Exception:
            # if any conversion fails, fall back to numpy below
            pass

    # Convert tvals/pvals to numpy/pandas as appropriate
    # Do not force truth-value checks on Series; keep as is
    # Compute z/t-values if not provided
    if tvals is None:
        # params and bse might be pandas Series or numpy arrays
        try:
            tvals = params / bse
        except Exception:
            tvals = np.asarray(params) / np.asarray(bse)
    # Compute p-values if not provided
    if pvals is None:
        try:
            from scipy import stats
            pvals = 2 * stats.norm.sf(np.abs(np.asarray(tvals)))
        except Exception:
            from math import erf, sqrt
            def _norm_cdf(x):
                return 0.5 * (1.0 + erf(x / sqrt(2.0)))
            tarr = np.asarray(tvals)
            pvals = 2 * (1.0 - np.array([_norm_cdf(abs(v)) for v in tarr]))

    # Ensure we have a conf_int; otherwise compute normal-approx 95% CI
    if conf_int is None:
        ci_lower = np.asarray(params) - 1.96 * np.asarray(bse)
        ci_upper = np.asarray(params) + 1.96 * np.asarray(bse)
        if pd is not None:
            try:
                conf_int = pd.DataFrame({"2.5%": ci_lower, "97.5%": ci_upper}, index=getattr(params, "index", None))
            except Exception:
                conf_int = np.vstack([ci_lower, ci_upper]).T
        else:
            conf_int = np.vstack([ci_lower, ci_upper]).T

    # Build the results dictionary for fixed effects
    results = {}
    # determine names from params
    if pd is not None and isinstance(params, pd.Series):
        names = list(params.index)
    else:
        # fallback: numeric indices
        try:
            length = len(params)
        except Exception:
            length = len(np.asarray(params))
        names = [f"param_{i}" for i in range(length)]

    for i, name in enumerate(names):
        est = _get_value(params, name, i)
        se = _get_value(bse, name, i)
        tv = _get_value(tvals, name, i)
        pv = _get_value(pvals, name, i)

        # Extract CI
        ci_low = None
        ci_high = None
        try:
            # pandas DataFrame with index
            if pd is not None and isinstance(conf_int, pd.DataFrame) and name in conf_int.index:
                row = conf_int.loc[name]
                # row may be Series with two elements; take first and second
                try:
                    ci_vals = np.asarray(row)
                    ci_low, ci_high = float(ci_vals[0]), float(ci_vals[1])
                except Exception:
                    # row might be scalar if conf_int had single column; handle gracefully
                    ci_low = float(row)
                    ci_high = float(row)
            else:
                # conf_int as numpy-like
                arr = np.asarray(conf_int)
                ci_low, ci_high = float(arr[i, 0]), float(arr[i, 1])
        except Exception:
            # as a last resort, compute from est +/- 1.96*se
            try:
                ci_low = float(est) - 1.96 * float(se)
                ci_high = float(est) + 1.96 * float(se)
            except Exception:
                ci_low, ci_high = (None, None)

        # convert to floats safely
        try:
            est_f = float(est)
        except Exception:
            est_f = float(np.asarray(est).astype(float).item()) if est is not None else None
        try:
            se_f = float(se)
        except Exception:
            se_f = float(np.asarray(se).astype(float).item()) if se is not None else None
        try:
            tv_f = float(tv)
        except Exception:
            tv_f = float(np.asarray(tv).astype(float).item()) if tv is not None else None
        try:
            pv_f = float(pv)
        except Exception:
            pv_f = float(np.asarray(pv).astype(float).item()) if pv is not None else None

        # exponentiated effect (multiplicative change in the rate per unit increase of the predictor)
        try:
            exp_est = float(np.exp(est_f))
        except Exception:
            exp_est = None
        try:
            exp_ci_l = float(np.exp(ci_low)) if ci_low is not None else None
            exp_ci_u = float(np.exp(ci_high)) if ci_high is not None else None
        except Exception:
            exp_ci_l, exp_ci_u = (None, None)

        results[name] = {
            "estimate": est_f,
            "std_error": se_f,
            "t_or_z": tv_f,
            "p_value": pv_f,
            "ci_2.5%": float(ci_low) if ci_low is not None else None,
            "ci_97.5%": float(ci_high) if ci_high is not None else None,
            "exp_estimate": exp_est,
            "exp_ci_2.5%": exp_ci_l,
            "exp_ci_97.5%": exp_ci_u,
        }

    # Attempt to get variance components (random intercept variance and residual variance)
    var_components = {}
    try:
        cov_re = _safe_get_attr(model_output, "cov_re")
        if cov_re is not None:
            try:
                arr = np.asarray(cov_re)
                # If matrix, take first diagonal element (random intercept variance)
                if arr.size == 1:
                    var_components["random_intercept_variance"] = float(arr.ravel()[0])
                else:
                    # if covariance matrix, take [0,0]
                    var_components["random_intercept_variance"] = float(arr[0, 0])
            except Exception:
                try:
                    # pandas DataFrame/Series
                    if pd is not None and hasattr(cov_re, "values"):
                        var_components["random_intercept_variance"] = float(np.asarray(cov_re).ravel()[0])
                except Exception:
                    pass
    except Exception:
        pass

    try:
        scale = _safe_get_attr(model_output, "scale")
        if scale is not None:
            var_components["residual_variance"] = float(scale)
    except Exception:
        pass

    # Compose human-readable description focusing on the three predictors of interest
    def _interpret_term(name, info):
        # name mapping for clarity
        pretty = name
        if name == "age":
            pretty = "Age (years)"
            unit = "per year"
        elif name == "sex_male":
            pretty = "Sex (male vs female)"
            unit = "male vs female"
        elif name == "help_received":
            pretty = "Help received (yes vs no)"
            unit = "help vs no help"
        else:
            unit = ""
        est = info.get("estimate")
        pv = info.get("p_value")
        ci_l = info.get("ci_2.5%")
        ci_u = info.get("ci_97.5%")
        exp_est = info.get("exp_estimate")
        exp_l = info.get("exp_ci_2.5%")
        exp_u = info.get("exp_ci_97.5%")

        sig_text = "statistically significant" if (pv is not None and pv < 0.05) else "not statistically significant"
        # interpret direction
        if est is None:
            direction = "no estimate available"
        elif est > 0:
            direction = "associated with higher nut-cracking efficiency"
        elif est < 0:
            direction = "associated with lower nut-cracking efficiency"
        else:
            direction = "no directional association with nut-cracking efficiency"

        # Safely format numbers
        try:
            est_s = f"{est:.4f}"
        except Exception:
            est_s = str(est)
        try:
            ci_l_s = f"{ci_l:.4f}"
            ci_u_s = f"{ci_u:.4f}"
        except Exception:
            ci_l_s = str(ci_l)
            ci_u_s = str(ci_u)
        try:
            pv_s = f"{pv:.3g}"
        except Exception:
            pv_s = str(pv)
        try:
            exp_s = f"{exp_est:.3f}"
            exp_l_s = f"{exp_l:.3f}"
            exp_u_s = f"{exp_u:.3f}"
        except Exception:
            exp_s = str(exp_est)
            exp_l_s = str(exp_l)
            exp_u_s = str(exp_u)

        return (f"{pretty}: estimate={est_s}, 95% CI [{ci_l_s}, {ci_u_s}], p={pv_s}. "
                f"This is {sig_text}. Direction: {direction}. "
                f"On the multiplicative scale, exp(estimate)={exp_s} "
                f"(95% CI [{exp_l_s}, {exp_u_s}]) — e.g. a value >1 indicates a multiplicative increase "
                f"in the rate of nuts opened per second; <1 indicates a decrease.")

    # Build description for the three predictors if present
    desc_lines = []
    for var in ["age", "sex_male", "help_received"]:
        if var in results:
            desc_lines.append(_interpret_term(var, results[var]))
        else:
            desc_lines.append(f"{var}: not estimated / not present in model output.")

    # Combine into final return object
    final_obj = {
        "fixed_effects": results,
        "variance_components": var_components,
    }

    description = ("Extracted fixed-effect estimates (estimate, SE, t/z, p-value, 95% CI) and exponentiated "
                   "effects for interpretability. Below are brief interpretive statements for the three "
                   "predictors of interest:\n- " + "\n- ".join(desc_lines) +
                   "\n\nNotes: Coefficients are on the log-rate scale where log_rate = log((nuts_opened + 0.5)/seconds). "
                   "A positive coefficient means higher log-rate (more nuts opened per second). The exponentiated "
                   "estimate gives the multiplicative change in the raw rate (nuts/sec + small offset) per unit change "
                   "in the predictor. Statistical significance is assessed at alpha=0.05 using the reported p-values.")

    return {"object": final_obj, "description": description}