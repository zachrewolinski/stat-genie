def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels GLMResultsWrapper (cluster-robust results
    object returned by get_robustcov_results or the original fit) and summarize effects
    relevant to the task:
      - size_diff_z (relative group size)
      - home_advantage_z (contest location relative to home ranges)

    Returns a dictionary with:
      - "object": dict containing per-predictor statistics (coef, robust se, z, p, 95% CI,
                  odds ratio and 95% CI) and boolean significance flags (alpha=0.05)
      - "description": brief interpretation of the extracted results and how to read them.

    The function is defensive: it will try to access params, bse, pvalues, conf_int from the
    provided results object and will raise a clear error if those aren't present.
    """
    import numpy as np

    # Predictors of interest (including the intercept for completeness)
    predictors = ['const', 'size_diff_z', 'home_advantage_z', 'm_diff_z', 'total_size_z']

    # Helper to raise a clear error if attributes are missing
    def _ensure_attr(obj, attr):
        if not hasattr(obj, attr):
            raise AttributeError(f"The provided model_output has no attribute '{attr}'. "
                                 "Expected a statsmodels results-like object (params, bse, pvalues, conf_int).")

    # Some results wrappers (robustcov results) expose these attributes; ensure presence
    _ensure_attr(model_output, 'params')
    _ensure_attr(model_output, 'bse')
    _ensure_attr(model_output, 'pvalues')
    _ensure_attr(model_output, 'conf_int')  # method or attribute

    params = model_output.params
    bse = model_output.bse
    pvalues = model_output.pvalues

    # conf_int may be a method or attribute
    try:
        conf = model_output.conf_int()
    except TypeError:
        # If conf_int is an attribute (e.g., array-like)
        conf = model_output.conf_int

    # conf is expected as an array-like with shape (k, 2) and rows corresponding to params index order.
    # We'll map by index/label when possible.
    ci_lower = {}
    ci_upper = {}
    try:
        # If conf is a DataFrame-like with index matching params
        idx = list(params.index)
        for i, name in enumerate(idx):
            ci_lower[name] = float(conf[i, 0])
            ci_upper[name] = float(conf[i, 1])
    except Exception:
        # Fallback: try to index by label if conf supports .loc
        try:
            for name in params.index:
                row = conf.loc[name]
                ci_lower[name] = float(row[0])
                ci_upper[name] = float(row[1])
        except Exception:
            raise RuntimeError("Could not parse conf_int output; unexpected format.")

    results = {}
    alpha = 0.05
    for name in predictors:
        if name not in params.index:
            # Predictor missing from model (e.g., no const) — note as None
            results[name] = {
                'coef': None,
                'se': None,
                'z': None,
                'pvalue': None,
                'ci_lower': None,
                'ci_upper': None,
                'odds_ratio': None,
                'or_ci_lower': None,
                'or_ci_upper': None,
                'significant': None,
                'direction': None
            }
            continue

        coef = float(params[name])
        se = float(bse[name])
        # z (Wald) statistic for GLM approximate inference
        z_stat = coef / se if se != 0 else None
        pval = float(pvalues[name])

        lower = ci_lower.get(name, None)
        upper = ci_upper.get(name, None)

        # odds ratio and CI (exponentiate)
        or_val = float(np.exp(coef)) if coef is not None else None
        or_lower = float(np.exp(lower)) if lower is not None else None
        or_upper = float(np.exp(upper)) if upper is not None else None

        significant = None if pval is None else (pval < alpha)
        if coef is None:
            direction = None
        else:
            if coef > 0:
                direction = "positive (higher predictor -> higher log-odds of focal winning)"
            elif coef < 0:
                direction = "negative (higher predictor -> lower log-odds of focal winning)"
            else:
                direction = "null (no direction)"

        results[name] = {
            'coef': coef,
            'se': se,
            'z': z_stat,
            'pvalue': pval,
            'ci_lower': lower,
            'ci_upper': upper,
            'odds_ratio': or_val,
            'or_ci_lower': or_lower,
            'or_ci_upper': or_upper,
            'significant': significant,
            'direction': direction
        }

    # Construct a short interpretation string focusing on the two main predictors
    def interpret_pred(name):
        r = results.get(name)
        if r is None or r['coef'] is None:
            return f"{name}: not in model / no estimate available."
        sig_text = "statistically significant (p < 0.05)" if r['significant'] else "not statistically significant (p >= 0.05)"
        return (f"{name}: coef={r['coef']:.3f}, se={r['se']:.3f}, p={r['pvalue']:.3g}; "
                f"OR={r['odds_ratio']:.3f} (95% CI {r['or_ci_lower']:.3f}–{r['or_ci_upper']:.3f}); {sig_text}; "
                f"direction: {r['direction']}")

    description = (
        "Extracted per-predictor statistics (cluster-robust where provided). Interpretation for main predictors:\n"
        f"- size_diff_z (relative group size): {interpret_pred('size_diff_z')}\n"
        f"- home_advantage_z (contest location/home advantage): {interpret_pred('home_advantage_z')}\n\n"
        "How to read results: coef is the log-odds effect of a one-unit increase in the (standardized) predictor. "
        "Odds ratio (OR) = exp(coef) >1 implies higher odds of the focal group winning as the predictor increases; "
        "OR <1 implies lower odds. Significance is judged at alpha=0.05 using the provided (cluster-robust) p-values."
    )

    return {'object': results, 'description': description}