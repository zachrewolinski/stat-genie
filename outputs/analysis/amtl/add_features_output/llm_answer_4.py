def extract_final_answer(model_output):
    """
    Extract the effect of the IsHuman indicator from a fitted statsmodels results object
    (e.g., GLMResultsWrapper or the robust-covariance wrapper returned by
    get_robustcov_results).

    Returns a dictionary with:
      - "object": dict with numeric results for the IsHuman coefficient (coef, se, p, CI,
                  odds ratio and its CI, boolean for significance at alpha=0.05)
      - "description": short plain-language interpretation answering whether modern
                       humans have higher AMTL after controlling for covariates.
    """
    import numpy as np
    from math import erf, sqrt

    res = model_output

    # Get parameter series
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("Provided model_output has no 'params' attribute.")

    # Find the parameter name for IsHuman (allow for slight naming differences)
    candidate_keys = [k for k in params.index if k == "IsHuman" or k.startswith("IsHuman")]
    if not candidate_keys:
        # As a last resort, try exact match of common alternatives
        alt_keys = [k for k in params.index if "IsHuman" in k.replace(":", "_")]
        if alt_keys:
            candidate_keys = alt_keys

    if not candidate_keys:
        raise KeyError(f"Could not find a parameter corresponding to 'IsHuman' in model params: {list(params.index)}")

    key = candidate_keys[0]

    # Extract coefficient
    coef = float(params.loc[key])

    # Extract standard error: prefer res.bse, fallback to sqrt(diag(cov_params()))
    bse = None
    bse_attr = getattr(res, "bse", None)
    if bse_attr is not None:
        try:
            bse = float(bse_attr.loc[key])
        except Exception:
            bse = None
    if bse is None:
        # try covariance matrix
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = None
        if cov is not None:
            try:
                bse = float(np.sqrt(np.asarray(cov.loc[key, key], dtype=float)))
            except Exception:
                bse = None
    if bse is None:
        raise RuntimeError("Could not obtain standard error for the IsHuman parameter.")

    # Extract p-value: prefer res.pvalues, fallback to normal approximation using z-score
    pval = None
    pvals_attr = getattr(res, "pvalues", None)
    if pvals_attr is not None:
        try:
            pval = float(pvals_attr.loc[key])
        except Exception:
            pval = None
    if pval is None:
        # normal approximation
        z = coef / bse
        # standard normal cdf using erf
        def std_norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))
        pval = 2.0 * (1.0 - std_norm_cdf(abs(z)))

    # Confidence interval: prefer res.conf_int(), fallback to coef +/- 1.96*se
    ci = None
    try:
        ci_df = res.conf_int()
        # conf_int may return a DataFrame or ndarray; try to access by label
        if hasattr(ci_df, "loc"):
            ci_vals = ci_df.loc[key].astype(float).values
        else:
            # if it is ndarray, find index of key in params to index into rows
            idx = list(params.index).index(key)
            ci_vals = np.asarray(ci_df, dtype=float)[idx, :]
        ci = np.asarray(ci_vals, dtype=float)
    except Exception:
        ci = np.array([coef - 1.96 * bse, coef + 1.96 * bse], dtype=float)

    # Convert to odds ratio scale (logit link => exponentiate)
    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = list(np.exp(ci).astype(float))

    significant = bool(pval < 0.05)

    # Build concise interpretation
    if coef > 0 and significant:
        interpretation = (
            f"Yes — after controlling for age, prob_male, and tooth class, the IsHuman coefficient "
            f"is positive and statistically significant (coef = {coef:.3f}, SE = {bse:.3f}, p = {pval:.3g}). "
            f"In odds-ratio terms this is OR = {odds_ratio:.3f} (95% CI {odds_ratio_ci[0]:.3f}–{odds_ratio_ci[1]:.3f}), "
            "indicating higher AMTL frequency in modern humans."
        )
    elif coef > 0 and not significant:
        interpretation = (
            f"No strong evidence that humans have higher AMTL after controls: coefficient is positive "
            f"but not statistically significant (coef = {coef:.3f}, SE = {bse:.3f}, p = {pval:.3g}). "
            f"OR = {odds_ratio:.3f} (95% CI {odds_ratio_ci[0]:.3f}–{odds_ratio_ci[1]:.3f})."
        )
    elif coef < 0 and significant:
        interpretation = (
            f"No — modern humans have significantly lower AMTL after controls (coef = {coef:.3f}, "
            f"SE = {bse:.3f}, p = {pval:.3g}). OR = {odds_ratio:.3f} (95% CI {odds_ratio_ci[0]:.3f}–{odds_ratio_ci[1]:.3f})."
        )
    else:
        interpretation = (
            f"No evidence of a difference in AMTL between humans and non-human primates after controls "
            f"(coef = {coef:.3f}, SE = {bse:.3f}, p = {pval:.3g}). OR = {odds_ratio:.3f} (95% CI {odds_ratio_ci[0]:.3f}–{odds_ratio_ci[1]:.3f})."
        )

    result_object = {
        "parameter_name": key,
        "coef": coef,
        "std_err": bse,
        "p_value": pval,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": odds_ratio_ci,
        "significant_at_0.05": significant,
    }

    return {"object": result_object, "description": interpretation}