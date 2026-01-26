def extract_final_answer(model_output):
    """
    Extracts statistics related to the effect of instructor beauty on course evaluations
    from a fitted statsmodels RegressionResultsWrapper.

    Returns a dict:
      {
        "object": { ... detailed numbers ... },
        "description": "short explanation of what these numbers mean"
      }

    The function extracts:
      - Coefficients, standard errors, p-values and 95% CIs for Beauty, Beauty_sq, and the Beauty:Female interaction.
      - Marginal effects (derivative of Eval w.r.t. Beauty) at Beauty = 0 (mean-shifted) for males and females,
        with SE, z/t, p-value (normal approx) and 95% CI.
      - If the original data frame is available from the model, also reports marginal effects at ±1 SD of Beauty.
    """
    import numpy as np
    import pandas as pd
    from math import sqrt
    try:
        from scipy import stats
        _norm_sf = lambda x: stats.norm.sf(x)  # survival function
    except Exception:
        # fallback normal cdf using math.erf if scipy not available
        from math import erf
        def _norm_sf(x):
            return 0.5 * (1.0 - erf(x / sqrt(2.0)))

    res = model_output

    # Gather parameter names and values
    params = res.params
    cov = res.cov_params()  # robust (clustered) covariance if model was fit that way
    pvalues = res.pvalues
    conf_int = res.conf_int(alpha=0.05)  # DataFrame with 0,1 columns

    # Helper to find parameter name robustly
    def find_param_name(possible_names):
        # possible_names: list of strings to try matching exactly
        for name in possible_names:
            if name in params.index:
                return name
        # fallback: find param that contains all parts (case sensitive)
        for name in params.index:
            parts = [p for p in possible_names if p]
            if all(part in name for part in parts):
                return name
        return None

    beauty_name = find_param_name(['Beauty'])
    beauty_sq_name = find_param_name(['Beauty_sq', 'Beauty^2', 'Beauty_sq'])
    # interaction could be 'Beauty:Female' or 'Beauty:Female' etc.
    interaction_name = None
    # try obvious
    for cand in ['Beauty:Female', 'Beauty:Female', 'Beauty:Female', 'Beauty*Female']:
        if cand in params.index:
            interaction_name = cand
            break
    if interaction_name is None:
        # search any param that includes both substrings
        for name in params.index:
            if ('Beauty' in name) and ('Female' in name):
                interaction_name = name
                break

    # Collect base coefficient info (if present)
    def collect_coef_info(name):
        if name is None or name not in params.index:
            return None
        return {
            'name': name,
            'coef': float(params[name]),
            'se': float(np.sqrt(cov.loc[name, name])) if name in cov.index else float(res.bse.get(name, np.nan)),
            'pval': float(pvalues.get(name, np.nan)),
            'ci_lower': float(conf_int.loc[name, 0]),
            'ci_upper': float(conf_int.loc[name, 1])
        }

    beauty_info = collect_coef_info(beauty_name)
    beauty_sq_info = collect_coef_info(beauty_sq_name)
    interaction_info = collect_coef_info(interaction_name)

    # Prepare function to compute linear combination variance and CI
    def lincomb_stats(coef_vector):
        """
        coef_vector: pandas Series or 1D numpy array aligned with params.index
        (i.e., length equal to number of params). Only entries for parameters used need be nonzero.
        Returns dict with estimate, se, z, p (2-sided normal), CI.
        """
        idx = params.index
        a = np.zeros(len(idx), dtype=float)
        # align by index
        for i, name in enumerate(idx):
            if name in coef_vector:
                a[i] = float(coef_vector[name])
        est = float(np.dot(a, params.values))
        # variance = a' Cov a
        try:
            var = float(a @ cov.values @ a)
        except Exception:
            # fallback to NaN
            var = float('nan')
        se = float(sqrt(var)) if var >= 0 and not np.isnan(var) else float('nan')
        z = est / se if se and not np.isnan(se) else float('nan')
        p_two = 2.0 * _norm_sf(abs(z)) if not np.isnan(z) else float('nan')
        ci_low = est - 1.96 * se if not np.isnan(se) else float('nan')
        ci_high = est + 1.96 * se if not np.isnan(se) else float('nan')
        return {'estimate': est, 'se': se, 'z': z, 'p_value': p_two, 'ci_95': (ci_low, ci_high)}

    # Marginal effect (derivative) formulas:
    # dEval/dBeauty = beta_Beauty + 2 * beta_Beauty_sq * Beauty + beta_interaction * Female
    # We'll evaluate at Beauty = 0 (mean-shifted) for Male (Female=0) and Female (Female=1).
    lincomb_results = {}

    # Prepare zero-based coefficient vector names for convenience
    param_names = list(params.index)

    # Evaluate at Beauty = 0 (mean)
    b0 = 0.0
    # Male (Female = 0)
    coef_vec_male = {}
    if beauty_name in param_names:
        coef_vec_male[beauty_name] = 1.0
    if beauty_sq_name in param_names:
        coef_vec_male[beauty_sq_name] = 2.0 * b0
    if interaction_name in param_names:
        coef_vec_male[interaction_name] = 0.0
    lincomb_results['marginal_at_mean_male'] = lincomb_stats(coef_vec_male)

    # Female (Female = 1)
    coef_vec_female = {}
    if beauty_name in param_names:
        coef_vec_female[beauty_name] = 1.0
    if beauty_sq_name in param_names:
        coef_vec_female[beauty_sq_name] = 2.0 * b0
    if interaction_name in param_names:
        coef_vec_female[interaction_name] = 1.0
    lincomb_results['marginal_at_mean_female'] = lincomb_stats(coef_vec_female)

    # If original data available, compute ±1 SD points
    beauty_sd = None
    beauty_min = None
    beauty_max = None
    data_available = False
    try:
        df = res.model.data.frame  # this exists when formula API used and a dataframe was passed
        if beauty_name in df.columns:
            beauty_series = df[beauty_name].dropna().astype(float)
            if len(beauty_series) > 1:
                beauty_sd = float(beauty_series.std(ddof=1))
                beauty_min = float(beauty_series.min())
                beauty_max = float(beauty_series.max())
                data_available = True
    except Exception:
        data_available = False

    if data_available and (beauty_sd is not None) and beauty_sd > 0:
        for sign, label in [(-1, 'minus1sd'), (+1, 'plus1sd')]:
            bval = sign * beauty_sd
            # male
            coef_vec_m = {}
            if beauty_name in param_names:
                coef_vec_m[beauty_name] = 1.0
            if beauty_sq_name in param_names:
                coef_vec_m[beauty_sq_name] = 2.0 * bval
            if interaction_name in param_names:
                coef_vec_m[interaction_name] = 0.0
            lincomb_results[f'marginal_{label}_male'] = {'beauty_value': bval, **lincomb_stats(coef_vec_m)}

            # female
            coef_vec_f = {}
            if beauty_name in param_names:
                coef_vec_f[beauty_name] = 1.0
            if beauty_sq_name in param_names:
                coef_vec_f[beauty_sq_name] = 2.0 * bval
            if interaction_name in param_names:
                coef_vec_f[interaction_name] = 1.0
            lincomb_results[f'marginal_{label}_female'] = {'beauty_value': bval, **lincomb_stats(coef_vec_f)}
    else:
        # If data not available, provide marginal effect at +/-1 (interpretable if beauty is standardized; else skip)
        # We'll attempt +/-1 unit as a fallback (but note: Beauty is mean-shifted; user should prefer actual SD if available).
        for bval, lab in [(-1.0, 'minus1'), (+1.0, 'plus1')]:
            coef_vec_m = {}
            if beauty_name in param_names:
                coef_vec_m[beauty_name] = 1.0
            if beauty_sq_name in param_names:
                coef_vec_m[beauty_sq_name] = 2.0 * bval
            if interaction_name in param_names:
                coef_vec_m[interaction_name] = 0.0
            lincomb_results[f'marginal_{lab}_male'] = {'beauty_value': bval, **lincomb_stats(coef_vec_m)}

            coef_vec_f = {}
            if beauty_name in param_names:
                coef_vec_f[beauty_name] = 1.0
            if beauty_sq_name in param_names:
                coef_vec_f[beauty_sq_name] = 2.0 * bval
            if interaction_name in param_names:
                coef_vec_f[interaction_name] = 1.0
            lincomb_results[f'marginal_{lab}_female'] = {'beauty_value': bval, **lincomb_stats(coef_vec_f)}

    # Summarize whether the marginal effects at mean are statistically different from zero (alpha=0.05)
    def significance_label(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 'unknown'
        if p < 0.01:
            return 'p < 0.01'
        if p < 0.05:
            return 'p < 0.05'
        if p < 0.10:
            return 'p < 0.10'
        return 'not significant (p >= 0.10)'

    summary_flags = {
        'marginal_at_mean_male_significance': significance_label(lincomb_results['marginal_at_mean_male']['p_value']),
        'marginal_at_mean_female_significance': significance_label(lincomb_results['marginal_at_mean_female']['p_value'])
    }

    # Build final object
    output_object = {
        'coefficients': {
            'Beauty': beauty_info,
            'Beauty_sq': beauty_sq_info,
            'Beauty:Female_interaction': interaction_info
        },
        'marginal_effects': lincomb_results,
        'significance_summary': summary_flags,
        # include some metadata
        'notes': (
            "Marginal effects are d(Eval)/d(Beauty) = beta_Beauty + 2*beta_Beauty_sq*Beauty + beta_interaction*Female. "
            "Estimates use the model's covariance matrix (cluster-robust if the model was fit that way). "
            "P-values for linear combinations are computed using a normal approximation (two-sided)."
        )
    }

    description = (
        "This output reports the estimated coefficients (coef, SE, p-value, 95% CI) for the linear, quadratic, "
        "and interaction terms involving instructor beauty, plus the marginal effect of beauty on evaluations "
        "for males and females evaluated at Beauty = 0 (the mean, since Beauty is mean-shifted). "
        "If the model's original data frame was available, marginal effects are also shown at ±1 SD of Beauty; "
        "otherwise the function reports marginal effects at ±1 unit as a fallback. "
        "Significance of the marginal effects at the mean is summarized in 'significance_summary'."
    )

    return {"object": output_object, "description": description}