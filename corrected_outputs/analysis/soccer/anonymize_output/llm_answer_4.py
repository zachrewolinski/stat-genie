def extract_final_answer(model_output):
    """
    Extracts the estimated effect of SkinDark from a fitted statsmodels GLMResults-like object,
    computes incidence rate ratio (IRR) and 95% CI for the IRR, and gives a short interpretation.

    Returns a dict with keys:
      - "object": a dict with numeric results (coef, se, p-value, CI, IRR, nobs, cov_type)
      - "description": a short plain-language interpretation of what the numbers imply
    """
    import numpy as np
    import re

    res = model_output

    # Helper to safely get an attribute or raise a helpful error
    def _get_attr(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    # Get parameter series (params)
    params = _get_attr(res, "params")
    if params is None:
        # some wrappers store results in ._results or .results
        params = _get_attr(getattr(res, "_results", None), "params") or _get_attr(getattr(res, "results", None), "params")
    if params is None:
        raise ValueError("Could not find parameter estimates (res.params) in model_output.")

    # Find the parameter name corresponding to SkinDark (robust to slight renamings)
    param_names = list(params.index) if hasattr(params, "index") else list(params.keys())
    skin_names = [n for n in param_names if re.search(r"\bSkinDark\b", n)]
    if not skin_names:
        # fallback: any parameter containing the substring
        skin_names = [n for n in param_names if "SkinDark" in n]
    if not skin_names:
        raise ValueError(f"Could not find a parameter for 'SkinDark'. Available parameters: {param_names}")

    param_name = skin_names[0]

    # Extract coefficient
    coef = float(params[param_name])

    # Standard error: try res.bse, else compute from cov_params()
    bse = None
    try:
        bse_attr = getattr(res, "bse", None)
        if bse_attr is not None and param_name in bse_attr:
            bse = float(bse_attr[param_name])
    except Exception:
        bse = None

    if bse is None:
        # try cov_params
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            # some wrappers keep cov in ._results
            try:
                cov = res._results.cov_params()
            except Exception:
                cov = None
        if cov is not None:
            # cov may be DataFrame or ndarray
            try:
                if hasattr(cov, "index"):
                    # DataFrame-like
                    idx = list(cov.index).index(param_name)
                    bse = float(np.sqrt(np.diag(cov))[idx])
                else:
                    # ndarray: need to find index of param in param_names
                    idx = param_names.index(param_name)
                    bse = float(np.sqrt(np.diag(cov))[idx])
            except Exception:
                bse = None

    if bse is None:
        # last resort: try attribute access directly
        try:
            bse = float(res.bse[param_name])
        except Exception:
            bse = np.nan

    # p-value
    pvalue = np.nan
    try:
        pvals_attr = getattr(res, "pvalues", None)
        if pvals_attr is not None and param_name in pvals_attr:
            pvalue = float(pvals_attr[param_name])
        else:
            # try to compute z and p from coef and se if available (approx)
            if not np.isnan(bse) and bse != 0:
                z = coef / bse
                # two-sided p-value using error function for normal cdf
                pvalue = float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(z) / np.sqrt(2.0)))))
            else:
                pvalue = np.nan
    except Exception:
        pvalue = np.nan

    # Confidence interval for coefficient
    ci_lower = ci_upper = None
    try:
        ci = res.conf_int()
        # conf_int may be a DataFrame or ndarray
        if hasattr(ci, "loc") and param_name in ci.index:
            row = ci.loc[param_name]
            ci_lower = float(row[0])
            ci_upper = float(row[1])
        else:
            # try to find index
            if hasattr(ci, "index"):
                idx = list(ci.index).index(param_name)
                ci_lower = float(ci.iloc[idx, 0])
                ci_upper = float(ci.iloc[idx, 1])
            else:
                # ndarray fallback: assume same order as params
                idx = param_names.index(param_name)
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
    except Exception:
        # fallback using normal approximation
        if not np.isnan(bse):
            ci_lower = coef - 1.96 * bse
            ci_upper = coef + 1.96 * bse
        else:
            ci_lower = ci_upper = np.nan

    # Incidence Rate Ratio (IRR) and CI on IRR scale
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None and not np.isnan(ci_lower) else np.nan
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None and not np.isnan(ci_upper) else np.nan
    percent_change = (irr - 1.0) * 100.0  # percent change in rate for SkinDark=1 vs 0

    # number of observations
    nobs = None
    try:
        if hasattr(res, "nobs"):
            try:
                nobs = int(res.nobs)
            except Exception:
                try:
                    nobs = int(float(res.nobs))
                except Exception:
                    nobs = None
        if nobs is None:
            # try model.nobs
            model = getattr(res, "model", None)
            if model is not None:
                nobs_attr = getattr(model, "nobs", None)
                if nobs_attr is not None:
                    try:
                        nobs = int(nobs_attr)
                    except Exception:
                        try:
                            nobs = int(float(nobs_attr))
                        except Exception:
                            nobs = None
                else:
                    # try model.endog size or length
                    endog = getattr(model, "endog", None)
                    if endog is not None:
                        try:
                            # prefer .size, fallback to len()
                            if hasattr(endog, "size"):
                                nobs = int(endog.size)
                            else:
                                nobs = int(len(endog))
                        except Exception:
                            nobs = None
    except Exception:
        nobs = None

    # Covariance type if available (to indicate whether clustering/robust was used)
    cov_type = getattr(res, "cov_type", None)
    cov_kwds = getattr(res, "cov_kwds", None)

    # Interpret the result in plain language
    alpha = 0.05
    significant = (not np.isnan(pvalue)) and (pvalue < alpha)
    if significant and coef > 0:
        conclusion = (
            "Yes — the estimated coefficient for SkinDark is positive and statistically significant "
            f"(p = {pvalue:.3g}). The estimated IRR = {irr:.3f} (95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f}), "
            f"meaning players rated Dark are estimated to receive about {percent_change:.1f}% more red cards "
            "per match-exposure than players rated Light, controlling for included covariates."
        )
    elif significant and coef < 0:
        conclusion = (
            "No — the estimated coefficient for SkinDark is negative and statistically significant "
            f"(p = {pvalue:.3g}). The estimated IRR = {irr:.3f} (95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f}), "
            f"meaning players rated Dark are estimated to receive about {percent_change:.1f}% fewer red cards "
            "per match-exposure than players rated Light, controlling for included covariates."
        )
    else:
        # not significant
        # Specify direction if coef nonzero
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        conclusion = (
            "No statistically significant evidence of a difference. "
            f"The coefficient for SkinDark is {'positive' if coef > 0 else ('negative' if coef < 0 else 'zero')} "
            f"(coef = {coef:.4g}, p = {pvalue:.3g}), giving an IRR = {irr:.3f} "
            f"(95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f}). This is consistent with {direction} "
            "red-card rates for Dark vs Light players but not at conventional significance levels."
        )

    result_object = {
        "parameter_name": param_name,
        "coef": coef,
        "std_err": float(bse) if bse is not None else np.nan,
        "p_value": float(pvalue) if not np.isnan(pvalue) else np.nan,
        "95%_CI_coef": [float(ci_lower) if ci_lower is not None else np.nan,
                        float(ci_upper) if ci_upper is not None else np.nan],
        "IRR": irr,
        "95%_CI_IRR": [irr_ci_lower, irr_ci_upper],
        "percent_change_IRR": percent_change,
        "n_obs": nobs,
        "cov_type": cov_type,
        "cov_kwds": cov_kwds
    }

    description = (
        f"Model: negative binomial with log(Matches) as offset. Parameter extracted: '{param_name}'. "
        f"Coefficient = {coef:.4g}, SE ≈ {bse:.4g}, p = {pvalue:.3g}. IRR = {irr:.3f} "
        f"(95% CI: {irr_ci_lower:.3f} to {irr_ci_upper:.3f}). {conclusion} "
        f"(n = {nobs if nobs is not None else 'unknown'})."
    )

    return {"object": result_object, "description": description}