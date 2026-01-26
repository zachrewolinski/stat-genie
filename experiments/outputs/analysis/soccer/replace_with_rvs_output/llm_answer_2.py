def extract_final_answer(model_output):
    """
    Extracts the Estimated Effect of IsDark from the provided RobustResults-like object.

    Returns:
      {
        "object": {
            "param_name": str,
            "coef": float,               # log rate ratio (negative binomial)
            "se": float,
            "z": float,
            "p": float,
            "conf_int_log": (lo, hi),    # 95% CI on log scale
            "IRR": float,                # incidence rate ratio = exp(coef)
            "IRR_95CI": (irr_lo, irr_hi),
            "IRR_percent_change": float, # (IRR - 1) * 100
            "IRR_percent_change_95CI": (pct_lo, pct_hi)
        },
        "description": str
      }
    """
    import numpy as np
    from scipy import stats
    import pandas as pd

    # Helper to safely get attributes from either the wrapper or the original fit
    def _get_attr(obj, name):
        return getattr(obj, name, getattr(getattr(obj, "_orig", object()), name, None))

    # Try to obtain parameter vector and related stats
    params = _get_attr(model_output, "params")
    bse = _get_attr(model_output, "bse")
    pvalues = _get_attr(model_output, "pvalues")
    zvalues = _get_attr(model_output, "zvalues")
    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        cov = None

    # Convert to pandas Series when possible for consistent indexing
    if params is None:
        raise ValueError("Model output does not expose 'params'. Cannot extract results.")
    if not isinstance(params, (pd.Series,)):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Unable to coerce params to a pandas Series.")

    if bse is not None and not isinstance(bse, pd.Series):
        try:
            bse = pd.Series(bse, index=params.index[: len(bse)])
        except Exception:
            # fallback: compute from covariance if available
            bse = None

    if pvalues is not None and not isinstance(pvalues, pd.Series):
        try:
            pvalues = pd.Series(pvalues, index=params.index[: len(pvalues)])
        except Exception:
            pvalues = None

    if zvalues is not None and not isinstance(zvalues, pd.Series):
        try:
            zvalues = pd.Series(zvalues, index=params.index[: len(zvalues)])
        except Exception:
            zvalues = None

    # Find the parameter name corresponding to IsDark.
    # The model's variable was named 'IsDark' in the formula; however, encoding
    # or transformations might change the exact parameter name. Search for matches.
    param_candidates = [name for name in params.index if "IsDark" in str(name)]
    if len(param_candidates) == 0:
        # try common binary encodings
        for pat in ["IsDark[T.True]", "IsDark[T.1]", "IsDark_1", "IsDark:"]:
            param_candidates += [name for name in params.index if pat in str(name)]
    if len(param_candidates) == 0:
        raise ValueError("Could not locate a parameter corresponding to 'IsDark' in model params.")

    param_name = param_candidates[0]

    coef = float(params[param_name])

    # get se: prefer provided bse, otherwise try diag(cov)
    se = None
    if bse is not None and param_name in bse.index:
        se = float(bse[param_name])
    else:
        # try from covariance matrix if available
        try:
            cov_mat = np.asarray(cov)
            # align index if cov is a DataFrame with columns
            if hasattr(cov, "columns"):
                cols = list(cov.columns)
                if param_name in cols:
                    se = float(np.sqrt(cov_mat[cols.index(param_name), cols.index(param_name)]))
            else:
                # assume covariance matrix ordering matches params ordering
                idx = list(params.index).index(param_name)
                se = float(np.sqrt(cov_mat[idx, idx]))
        except Exception:
            se = None

    if se is None or se == 0:
        raise ValueError("Could not determine a standard error for parameter '{}'.".format(param_name))

    # z and p: prefer provided
    if zvalues is not None and param_name in zvalues.index:
        z = float(zvalues[param_name])
    else:
        z = coef / se

    if pvalues is not None and param_name in pvalues.index:
        p = float(pvalues[param_name])
    else:
        p = float(2 * stats.norm.sf(abs(z)))

    # 95% CI on log scale and exponentiate to get IRR CI
    z_crit = stats.norm.ppf(0.975)
    ci_lo_log = coef - z_crit * se
    ci_hi_log = coef + z_crit * se

    irr = float(np.exp(coef))
    irr_lo = float(np.exp(ci_lo_log))
    irr_hi = float(np.exp(ci_hi_log))

    irr_pct = (irr - 1.0) * 100.0
    irr_pct_lo = (irr_lo - 1.0) * 100.0
    irr_pct_hi = (irr_hi - 1.0) * 100.0

    # Interpret significance
    alpha = 0.05
    significant = (p < alpha)

    if significant:
        if irr > 1.0:
            verdict = (
                "Statistically significant: Dark-skinned players receive more red cards. "
                "Estimated IRR = {irr:.3f} (95% CI: {l:.3f}–{h:.3f}), p = {p:.3g}. "
                "This corresponds to an estimated {pct:.1f}% increase in red card rate per game "
                "from the referee for darker-skinned players, controlling for covariates."
            ).format(irr=irr, l=irr_lo, h=irr_hi, p=p, pct=irr_pct)
        else:
            verdict = (
                "Statistically significant: Dark-skinned players receive fewer red cards. "
                "Estimated IRR = {irr:.3f} (95% CI: {l:.3f}–{h:.3f}), p = {p:.3g}. "
                "This corresponds to an estimated {pct:.1f}% decrease in red card rate per game "
                "for darker-skinned players, controlling for covariates."
            ).format(irr=irr, l=irr_lo, h=irr_hi, p=p, pct=irr_pct)
    else:
        verdict = (
            "No statistically significant evidence of a difference (p = {p:.3g}). "
            "Estimated IRR = {irr:.3f} (95% CI: {l:.3f}–{h:.3f}), which corresponds to an estimated "
            "{pct:.1f}% change (95% CI: {pct_lo:.1f}% to {pct_hi:.1f}%). "
            "Conclusion: the point estimate suggests a {direction} effect, but it is not statistically significant "
            "at alpha = {alpha}."
        ).format(
            p=p,
            irr=irr,
            l=irr_lo,
            h=irr_hi,
            pct=irr_pct,
            pct_lo=irr_pct_lo,
            pct_hi=irr_pct_hi,
            direction=("higher" if irr > 1.0 else "lower"),
            alpha=alpha
        )

    result_obj = {
        "param_name": param_name,
        "coef": coef,
        "se": se,
        "z": z,
        "p": p,
        "conf_int_log": (ci_lo_log, ci_hi_log),
        "IRR": irr,
        "IRR_95CI": (irr_lo, irr_hi),
        "IRR_percent_change": irr_pct,
        "IRR_percent_change_95CI": (irr_pct_lo, irr_pct_hi),
        "significant": bool(significant)
    }

    description = (
        "Parameter '{}' is the model coefficient for the indicator comparing dark vs light skin. "
        "In a negative binomial model with log(games) offset this coefficient is a log rate-ratio. "
        "Below are the extracted statistics and a short interpretation:\n\n{}"
    ).format(param_name, verdict)

    return {"object": result_obj, "description": description}