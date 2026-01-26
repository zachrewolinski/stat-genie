def extract_final_answer(model_output):
    """
    Extracts the estimated effect(s) of instructor beauty (beauty_c) on eval
    from a fitted OLS model with clustered (by prof) robust SEs.

    Expects model_output to be the dict returned by the provided `model` function:
      {'ols': <RegressionResultsWrapper>, 'clustered': <OLSResults (robust cov)>}

    Returns:
      {
        "object": {
          "baseline_group": { "coef": ..., "se": ..., "t": ..., "p": ..., "ci95": [low, high] },
          "by_gender": {
              "<level_name>": { "coef": ..., "se": ..., "t": ..., "p": ..., "ci95": [low, high] },
              ...
          },
          "notes": "..."
        },
        "description": "Brief interpretation of these numbers in context."
      }
    """
    import re
    import math
    import numpy as np

    # Normal cdf for p-values (fallback if scipy not available)
    try:
        from scipy import stats
        _norm_cdf = stats.norm.cdf
    except Exception:
        def _norm_cdf(x):
            # use erf-based normal cdf
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # retrieve clustered results (robust cov results)
    if isinstance(model_output, dict) and 'clustered' in model_output:
        clustered = model_output['clustered']
    else:
        raise ValueError("model_output must be a dict containing key 'clustered' with the robust-result object.")

    # Obtain parameter names and values in a robust way (params may be pandas Series or numpy array)
    raw_params = clustered.params
    if hasattr(raw_params, 'index'):
        param_names = list(raw_params.index)
        param_values = np.asarray(raw_params, dtype=float)
    else:
        # params is likely a numpy array; try to pull names from the model object
        param_values = np.asarray(raw_params, dtype=float)
        try:
            param_names = list(clustered.model.exog_names)
        except Exception:
            raise KeyError("Parameter names not found. params is an array and model.exog_names unavailable.")

    if 'beauty_c' not in param_names:
        raise KeyError("Expected coefficient named 'beauty_c' in model parameters. Found: {}".format(param_names))

    # Build mapping name -> value
    params_map = {name: float(param_values[i]) for i, name in enumerate(param_names)}

    # Handle covariance: could be pandas DataFrame or numpy array
    cov_raw = clustered.cov_params()
    cov_is_df = hasattr(cov_raw, 'loc') and hasattr(cov_raw, 'values')

    if cov_is_df:
        # convert to numpy for numeric ops but keep names for lookup
        cov_array = np.asarray(cov_raw.values, dtype=float)
        cov_names = list(cov_raw.index)
        name_to_pos = {n: i for i, n in enumerate(cov_names)}
    else:
        cov_array = np.asarray(cov_raw, dtype=float)
        # assume ordering matches param_names
        if cov_array.shape[0] != len(param_names):
            # mismatch; still attempt to continue but lookups will return nan
            name_to_pos = {n: i for i, n in enumerate(param_names)}
        else:
            name_to_pos = {n: i for i, n in enumerate(param_names)}

    def get_cov(x, y):
        """Return covariance between parameters named x and y, or NaN if not available."""
        # try direct name lookup
        if x in name_to_pos and y in name_to_pos:
            i = name_to_pos[x]
            j = name_to_pos[y]
            try:
                return float(cov_array[i, j])
            except Exception:
                return float(np.nan)
        # if cov was DataFrame, try reversed or fallbacks using label matching
        if cov_is_df:
            # try possible alternative label orders
            for a, b in ((x, y), (y, x)):
                if a in cov_raw.index and b in cov_raw.columns:
                    try:
                        return float(cov_raw.loc[a, b])
                    except Exception:
                        continue
        return float(np.nan)

    # Helper to compute p-value and CI using normal approximation
    def summarize_linear_combination(coef, var):
        se = float(np.sqrt(var)) if (not np.isnan(var) and var >= 0) else float(np.nan)
        t_stat = float(coef / se) if (se and not math.isnan(se)) else float('nan')
        p_val = float(2.0 * (1.0 - _norm_cdf(abs(t_stat)))) if not math.isnan(t_stat) else float('nan')
        crit = 1.96  # 95% approx
        ci_low = float(coef - crit * se) if not math.isnan(se) else float('nan')
        ci_high = float(coef + crit * se) if not math.isnan(se) else float('nan')
        return {"coef": float(coef), "se": se, "t": t_stat, "p": p_val, "ci95": [ci_low, ci_high]}

    results = {}
    # Baseline (reference gender) effect = coef(beauty_c)
    beta_b = float(params_map['beauty_c'])
    var_b = get_cov('beauty_c', 'beauty_c')
    results['baseline_group'] = summarize_linear_combination(beta_b, var_b)

    # Find interaction terms involving beauty_c (beauty_c:C(gender)[T.<level>] typically)
    interaction_names = [n for n in param_names if ('beauty_c' in n and ':' in n)]

    by_gender = {}
    if interaction_names:
        # Attempt to parse level names from the parameter names; fallback to raw param name
        for iname in interaction_names:
            # get interaction coefficient
            beta_int = float(params_map.get(iname, float(np.nan)))
            # compute linear combination: beauty_c + interaction
            combo_coef = beta_b + beta_int

            # variance of sum: var(b) + var(int) + 2*cov(b,int)
            var_int = get_cov(iname, iname)
            cov_b_int = get_cov('beauty_c', iname)
            if not math.isnan(var_b) and not math.isnan(var_int) and not math.isnan(cov_b_int):
                combo_var = var_b + var_int + 2.0 * cov_b_int
            else:
                combo_var = float(np.nan)

            # parse readable level name
            # common pattern: 'beauty_c:C(gender)[T.male]' -> extract 'male'
            m = re.search(r"C\(gender\)\[T\.?([^\]]+)\]", iname)
            if m:
                level = m.group(1)
            else:
                # maybe ordering reversed 'C(gender)[T.male]:beauty_c' or other; try another pattern on reversed string
                rev = iname[::-1]
                m2 = re.search(r"]\[([^\[]+)\]\)redneg\(.C", rev)
                if m2:
                    level = m2.group(1)[::-1]
                else:
                    # more general attempt: find T.<level> anywhere
                    m3 = re.search(r"T\.?([A-Za-z0-9_ -]+)", iname)
                    if m3:
                        level = m3.group(1)
                    else:
                        level = iname  # fallback

            by_gender[level] = summarize_linear_combination(combo_coef, combo_var)

        results['by_gender'] = by_gender

        # Optionally compute difference between a non-reference level and baseline (same as interaction coef)
        raw_interactions = {iname: float(params_map.get(iname, float(np.nan))) for iname in interaction_names}
        results['raw_interaction_coefficients'] = raw_interactions
    else:
        results['by_gender'] = {}
        results['raw_interaction_coefficients'] = {}

    # Add some meta notes
    results['notes'] = (
        "Estimates show the marginal effect of a one-unit increase in mean-centered beauty (beauty_c) "
        "on the course evaluation score (eval, 1-5 scale). 'baseline_group' is the reference gender "
        "(the category omitted by C(gender) in the model). Entries under 'by_gender' give the marginal "
        "beauty effect for each non-reference gender category (i.e., baseline effect + interaction). "
        "Standard errors, t-statistics and p-values use the cluster-robust covariance matrix (clustered by prof). "
        "95% CIs are approximate (normal approx)."
    )

    # Short interpretation string based on p-values (if available)
    try:
        sig_lines = []
        base_p = results['baseline_group']['p']
        if not math.isnan(base_p) and base_p < 0.05:
            sig_lines.append("Beauty effect is statistically significant for the reference gender (p={:.3f}).".format(base_p))
        else:
            sig_lines.append("Beauty effect is NOT statistically significant for the reference gender (p={:.3f}).".format(base_p))
        for lvl, statsd in results['by_gender'].items():
            p = statsd['p']
            if not math.isnan(p) and p < 0.05:
                sig_lines.append("Beauty effect is statistically significant for gender '{}'(p={:.3f}).".format(lvl, p))
            else:
                sig_lines.append("Beauty effect is NOT statistically significant for gender '{}'(p={:.3f}).".format(lvl, p))
        results['significance_summary'] = " ".join(sig_lines)
    except Exception:
        results['significance_summary'] = "Could not form a significance summary."

    description = (
        "Returned object provides the estimated marginal effect(s) of beauty (beauty_c) on course evaluation (eval). "
        "Use baseline_group to see the effect for the reference gender and by_gender for each other gender level. "
        "Each entry contains coef (change in eval per 1 unit of mean-centered beauty), clustered-robust SE, t-stat, p-value, "
        "and approximate 95% CI. Interpret coefficients as the expected change in the evaluation score associated with a one-unit "
        "increase in beauty (after centering)."
    )

    return {"object": results, "description": description}