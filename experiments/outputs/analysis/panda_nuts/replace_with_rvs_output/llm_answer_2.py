def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-stats, p-values, 95% CIs, and a brief
    interpretation for the predictors of interest (age_c, sex_m, help_y) from a
    statsmodels MixedLMResultsWrapper (or similar).

    Returns a dictionary with:
      - "object": dict mapping extracted items (fixed effects, variances, counts)
      - "description": short textual interpretation of what those stats mean
    """
    # predictors of interest
    predictors = ['age_c', 'sex_m', 'help_y']

    # Prepare container for extracted stats
    results = {}

    # Helper to safely get an attribute/element
    def safe_get(mapping, key, default=None):
        if mapping is None:
            return default
        # If mapping is a pandas Series/DataFrame or dict-like
        try:
            return mapping[key]
        except Exception:
            pass
        try:
            return mapping.loc[key]
        except Exception:
            pass
        # If mapping is array-like and key is positional index
        try:
            if isinstance(key, int):
                return mapping[key]
        except Exception:
            pass
        return default

    # Try to extract common result tables/values
    try:
        params = model_output.params
    except Exception:
        raise ValueError("model_output has no attribute 'params' or is not a fitted statsmodels result.")

    # bse, pvalues, conf_int
    bse = getattr(model_output, 'bse', None)
    if bse is None:
        bse = getattr(model_output, 'std_errors', None)

    pvalues = getattr(model_output, 'pvalues', None)
    # conf_int() is a method returning a DataFrame/array
    try:
        ci_df = model_output.conf_int()
    except Exception:
        ci_df = None

    # Random-effects variance (if available) and residual scale
    re_var = None
    try:
        cov_re = getattr(model_output, 'cov_re', None)
        if cov_re is not None:
            # cov_re may be a DataFrame, Series, or ndarray
            try:
                # DataFrame-like
                if hasattr(cov_re, 'iloc'):
                    re_var = float(cov_re.iloc[0, 0])
                else:
                    # ndarray or nested list
                    re_var = float(cov_re[0][0])
            except Exception:
                # fallback: try to convert single-value containers
                try:
                    re_var = float(cov_re)
                except Exception:
                    re_var = None
    except Exception:
        re_var = None

    resid_var = None
    try:
        scale_val = getattr(model_output, 'scale', None)
        if scale_val is not None:
            resid_var = float(scale_val)
    except Exception:
        resid_var = None

    # Number of observations / groups
    nobs = getattr(model_output, 'nobs', None)
    # Try multiple ways to get number of groups
    ngroups = None
    try:
        # common location for group labels in MixedLM
        grp_labels = getattr(getattr(model_output, 'model', None), 'group_labels', None)
        if grp_labels is not None:
            try:
                ngroups = int(getattr(grp_labels, 'shape', (len(grp_labels),))[0])
            except Exception:
                try:
                    ngroups = int(len(grp_labels))
                except Exception:
                    ngroups = None
    except Exception:
        ngroups = None

    if ngroups is None:
        try:
            k_re = getattr(model_output, 'k_re', None)
            if k_re is not None:
                ngroups = int(k_re)
        except Exception:
            ngroups = None

    # Extract stats for each predictor
    for var in predictors:
        est = safe_get(params, var, None)
        se = safe_get(bse, var, None) if bse is not None else None
        p = safe_get(pvalues, var, None) if pvalues is not None else None

        # confidence intervals
        lower, upper = None, None
        if ci_df is not None:
            try:
                # Prefer label-based extraction
                row = None
                try:
                    row = ci_df.loc[var]
                except Exception:
                    # some ci outputs return a 2-column array without index
                    row = None
                if row is not None:
                    # row may be a Series/ndarray-like
                    if hasattr(row, 'values'):
                        vals = row.values
                    else:
                        vals = row
                    try:
                        lower = float(vals[0])
                        upper = float(vals[1])
                    except Exception:
                        lower, upper = None, None
                else:
                    # positional fallback based on params order
                    try:
                        idx = list(params.index).index(var)
                        lower = float(ci_df.iloc[idx, 0])
                        upper = float(ci_df.iloc[idx, 1])
                    except Exception:
                        lower, upper = None, None
            except Exception:
                lower, upper = None, None

        # z/t statistic if available (estimate / se)
        try:
            z = float(est / se) if (est is not None and se is not None and se != 0) else None
        except Exception:
            z = None

        # Significance summary
        sig = None
        try:
            if p is not None:
                pval = float(p)
                if pval < 0.001:
                    sig = 'p < 0.001'
                elif pval < 0.01:
                    sig = 'p < 0.01'
                elif pval < 0.05:
                    sig = 'p < 0.05'
                else:
                    sig = f'p = {pval:.3f} (ns)'
        except Exception:
            sig = None

        # Convert numeric-like values to floats where possible
        def to_float_or_none(x):
            try:
                return None if x is None else float(x)
            except Exception:
                return None

        results[var] = {
            'estimate': to_float_or_none(est),
            'std_error': to_float_or_none(se),
            'z_or_t': z,
            'p_value': to_float_or_none(p),
            'ci_95_lower': to_float_or_none(lower),
            'ci_95_upper': to_float_or_none(upper),
            'significance_summary': sig
        }

    # Build a brief interpretation string
    interpretations = []
    for var, stats in results.items():
        est = stats['estimate']
        p = stats['p_value']
        if est is None:
            interpretations.append(f"{var}: no estimate available.")
            continue
        direction = "positive" if est > 0 else ("zero" if est == 0 else "negative")
        if p is not None:
            if p < 0.05:
                interpretations.append(f"{var}: {direction} effect (estimate = {est:.3f}, {stats['significance_summary']}).")
            else:
                interpretations.append(f"{var}: {direction} effect (estimate = {est:.3f}) but not statistically significant ({stats['significance_summary']}).")
        else:
            interpretations.append(f"{var}: {direction} effect (estimate = {est:.3f}); p-value not available.")

    # Compose the final returned dictionary
    final_object = {
        'fixed_effects': results,
        'random_effects_variance_intercept': re_var,
        'residual_variance_scale': resid_var,
        'n_observations': int(nobs) if nobs is not None else None,
        'n_groups (chimpanzees)': int(ngroups) if ngroups is not None else None
    }

    description_lines = [
        "Extracted fixed-effect estimates (estimate, SE, z/t, p-value, 95% CI) for predictors: age_c, sex_m, help_y.",
        "Also includes random-intercept variance (if available), residual scale, and sample/group sizes.",
        "Brief interpretations:"
    ] + interpretations

    description = " ".join(description_lines)

    return {
        "object": final_object,
        "description": description
    }