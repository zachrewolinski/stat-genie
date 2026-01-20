def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, 95% CIs, and odds-ratios for the key predictors
    from a fitted statsmodels binary logit result object (as returned by the provided model()).

    Returns a dict with:
      - "object": a dict keyed by predictor name containing numeric results
      - "description": a short interpretation of the predictors of interest in the study context
    """
    import numpy as np

    # Get parameter names and estimates
    try:
        params = model_output.params  # pandas Series
    except Exception as e:
        raise ValueError("Could not read params from model_output") from e

    # Get robust standard errors, p-values, and confidence intervals
    # The model function overrides .bse, .pvalues, and .conf_int to return robust estimates.
    try:
        bse = getattr(model_output, 'bse')
    except Exception:
        bse = None

    try:
        pvalues = getattr(model_output, 'pvalues')
    except Exception:
        pvalues = None

    # conf_int may be a method or attribute depending on how it was overridden
    try:
        conf = model_output.conf_int() if callable(model_output.conf_int) else model_output.conf_int
    except Exception:
        conf = None

    # Ensure we have arrays/Series for bse and pvalues
    # If they are numpy arrays without index, align by params.index order
    names = list(params.index)

    # Convert params to plain numpy and names to list for indexing
    coef_vals = np.asarray(params.values, dtype=float)

    # Prepare containers
    bse_vals = None
    pvals_vals = None
    ci_lower = None
    ci_upper = None

    # Fill bse_vals
    if bse is None:
        # Try to compute from covariance if available
        try:
            cov = model_output.cov_params()
            bse_vals = np.sqrt(np.diag(cov))
        except Exception:
            bse_vals = np.full_like(coef_vals, np.nan)
    else:
        bse_arr = np.asarray(bse)
        if bse_arr.shape == coef_vals.shape:
            bse_vals = bse_arr.astype(float)
        else:
            # If bse is a pandas Series align by index
            try:
                bse_vals = np.asarray(bse[names], dtype=float)
            except Exception:
                bse_vals = np.asarray(bse, dtype=float)

    # Fill p-values
    if pvalues is None:
        # compute from z = coef / se if possible
        with np.errstate(divide='ignore', invalid='ignore'):
            z = coef_vals / bse_vals
            pvals_vals = 2 * (1 - 0.5 * (1 + np.erf(np.abs(z) / np.sqrt(2))))  # equivalent to 2*(1 - norm.cdf(|z|))
    else:
        p_arr = np.asarray(pvalues)
        if p_arr.shape == coef_vals.shape:
            pvals_vals = p_arr.astype(float)
        else:
            try:
                pvals_vals = np.asarray(pvalues[names], dtype=float)
            except Exception:
                pvals_vals = np.asarray(pvalues, dtype=float)

    # Fill confidence intervals
    if conf is None:
        # approximate using coef +/- 1.96*se
        z = 1.96
        ci_lower = coef_vals - z * bse_vals
        ci_upper = coef_vals + z * bse_vals
    else:
        conf_arr = np.asarray(conf)
        # conf could be (k,2) ndarray in the same order as params.index
        if conf_arr.shape[0] == coef_vals.shape[0] and conf_arr.shape[1] == 2:
            ci_lower = conf_arr[:, 0].astype(float)
            ci_upper = conf_arr[:, 1].astype(float)
        else:
            # If conf is a DataFrame-like with index, try to align by names
            try:
                ci_lower = np.asarray(conf[names].iloc[:, 0], dtype=float)
                ci_upper = np.asarray(conf[names].iloc[:, 1], dtype=float)
            except Exception:
                # fallback to approximate
                z = 1.96
                ci_lower = coef_vals - z * bse_vals
                ci_upper = coef_vals + z * bse_vals

    # Compute odds ratios and CIs
    or_vals = np.exp(coef_vals)
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)

    # Create result mapping for all parameters, but we'll emphasize the three of interest
    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "coef": float(coef_vals[i]) if np.isfinite(coef_vals[i]) else None,
            "se": float(bse_vals[i]) if (bse_vals is not None and np.isfinite(bse_vals[i])) else None,
            "p": float(pvals_vals[i]) if (pvals_vals is not None and np.isfinite(pvals_vals[i])) else None,
            "ci_95_lower": float(ci_lower[i]) if np.isfinite(ci_lower[i]) else None,
            "ci_95_upper": float(ci_upper[i]) if np.isfinite(ci_upper[i]) else None,
            "odds_ratio": float(or_vals[i]) if np.isfinite(or_vals[i]) else None,
            "or_95_lower": float(or_ci_lower[i]) if np.isfinite(or_ci_lower[i]) else None,
            "or_95_upper": float(or_ci_upper[i]) if np.isfinite(or_ci_upper[i]) else None,
            "significant_at_0.05": bool((pvals_vals[i] < 0.05) if (pvals_vals is not None and np.isfinite(pvals_vals[i])) else False)
        }

    # Focused interpretation for the question variables
    focus_vars = ['rel_size_z', 'location_adv_z', 'rel_size_x_location']
    interp_lines = []
    for v in focus_vars:
        if v in results:
            r = results[v]
            sig_text = "statistically significant (p < 0.05)" if r["significant_at_0.05"] else "not statistically significant (p >= 0.05)"
            direction = "positive" if (r["coef"] is not None and r["coef"] > 0) else ("negative" if (r["coef"] is not None and r["coef"] < 0) else "no clear")
            interp_lines.append(
                f"{v}: coef={r['coef']:.3f}, SE={r['se']:.3f}, p={r['p']:.3f}, 95% CI=[{r['ci_95_lower']:.3f}, {r['ci_95_upper']:.3f}], "
                f"OR={r['odds_ratio']:.3f} (95% CI [{r['or_95_lower']:.3f}, {r['or_95_upper']:.3f}]) -> {direction} effect on focal group's chance of winning, {sig_text}."
            )
        else:
            interp_lines.append(f"{v}: not present in model output.")

    description = (
        "Extracted logistic regression estimates for predictors of focal-group win probability. "
        "For each predictor we report coefficient (log-odds), robust SE, p-value, 95% CI, odds-ratio (OR) and OR 95% CI. "
        "Interpretation lines for the key variables follow:\n" + "\n".join(interp_lines)
        + "\n\nNotes: positive coefficient means the predictor increases the log-odds of the focal group winning; "
        "odds-ratio > 1 indicates higher odds of winning. The interaction term (rel_size_x_location) indicates whether the effect "
        "of relative group size changes with location advantage."
    )

    return {"object": results, "description": description}