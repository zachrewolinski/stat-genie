def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels logistic regression result
    (possibly with cluster-robust covariance via get_robustcov_results).

    Returns a dict with:
      - "object": dict mapping each predictor of interest to a dict of statistics:
          coef, se, z_or_t, pvalue, ci_lower, ci_upper, odds_ratio, or_ci_lower, or_ci_upper
      - "description": a concise interpretation of the results for the predictors
                       'log_size_ratio', 'rel_dist', and 'size_by_loc' (if present).
    """
    import math

    # Ensure we have expected attributes
    if not hasattr(model_output, "params"):
        raise ValueError("model_output doesn't have expected 'params' attribute.")

    # Predictors of interest for answering the task
    predictors = ['log_size_ratio', 'rel_dist', 'size_by_loc']

    params = model_output.params  # pandas Series
    # Try to obtain robust se / pvalues from the object; fall back to params and bse
    try:
        bse = model_output.bse
    except Exception:
        raise ValueError("model_output doesn't provide standard errors (bse).")

    # Try to get pvalues and conf_int; if missing, compute approximate values
    pvalues = None
    try:
        pvalues = model_output.pvalues
    except Exception:
        pvalues = None

    try:
        conf_int_df = model_output.conf_int()
        # conf_int returns DataFrame with two columns; ensure column names
        # conf_int_df.loc[var, 0], conf_int_df.loc[var, 1]
    except Exception:
        conf_int_df = None

    results = {}
    for var in predictors:
        if var not in params.index:
            # variable not in model; skip but note absence
            results[var] = {
                'present': False,
                'note': f"Variable '{var}' not found in model."
            }
            continue

        coef = float(params[var])
        se = float(bse[var])

        # compute z/t value from coef and se
        z_or_t = coef / se if se != 0 else float('nan')

        # p-value: prefer model_output.pvalues if available; otherwise approximate using normal
        if (pvalues is not None) and (var in pvalues.index):
            pval = float(pvalues[var])
        else:
            # two-sided p-value from standard normal
            try:
                # use erf to compute normal CDF without scipy
                pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_or_t) / math.sqrt(2.0))))
            except Exception:
                pval = float('nan')

        # Confidence interval: prefer conf_int() if available
        if (conf_int_df is not None) and (var in conf_int_df.index):
            ci_low = float(conf_int_df.loc[var].iloc[0])
            ci_high = float(conf_int_df.loc[var].iloc[1])
        else:
            # approximate 95% CI using normal approx
            z_crit = 1.96
            ci_low = coef - z_crit * se
            ci_high = coef + z_crit * se

        # Odds ratio and its CI
        try:
            or_point = math.exp(coef)
            or_ci_low = math.exp(ci_low)
            or_ci_high = math.exp(ci_high)
        except OverflowError:
            or_point = float('inf') if coef > 0 else 0.0
            or_ci_low = float('inf') if ci_low > 0 else 0.0
            or_ci_high = float('inf') if ci_high > 0 else 0.0

        results[var] = {
            'present': True,
            'coef': coef,
            'se': se,
            'z_or_t': z_or_t,
            'pvalue': pval,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'odds_ratio': or_point,
            'or_ci_lower': or_ci_low,
            'or_ci_upper': or_ci_high
        }

    # Build a concise interpretation/description
    lines = []
    alpha = 0.05
    for var in predictors:
        entry = results.get(var)
        if not entry:
            continue
        if not entry.get('present', False):
            lines.append(f"{var}: NOT IN MODEL ({entry.get('note')}).")
            continue

        coef = entry['coef']
        pval = entry['pvalue']
        or_point = entry['odds_ratio']
        or_ci_low = entry['or_ci_lower']
        or_ci_high = entry['or_ci_upper']

        signif = "statistically significant" if (pval is not None and pval < alpha) else "not statistically significant"
        direction = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        # Human-readable explanation tailored to variables
        if var == 'log_size_ratio':
            meaning = ("A positive coefficient means that as the focal group's size advantage "
                       "(log ratio focal/other) increases, the focal group's probability of winning increases.")
        elif var == 'rel_dist':
            meaning = ("A positive coefficient means that when the focal group is relatively closer to its home "
                       "(rel_dist > 0), its probability of winning increases.")
        else:  # size_by_loc
            meaning = ("This term (size_by_loc) likely captures how the effect of relative size varies with location "
                       "(i.e., an interaction or size-by-location effect). A positive coefficient implies that the "
                       "benefit of being numerically larger is greater when the focal group has the location advantage.")

        lines.append(
            f"{var}: coef={coef:.3f}, SE={entry['se']:.3f}, p={pval:.3g} -> {signif}; "
            f"direction={direction}. Odds ratio={or_point:.3f} (95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}]). "
            f"Interpretation: {meaning}"
        )

    description = " | ".join(lines) if lines else "No predictors of interest found in the model."

    return {
        "object": results,
        "description": description
    }