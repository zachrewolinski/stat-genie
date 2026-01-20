def extract_final_answer(model_output):
    """
    Extracts age-related slopes (developmental change per year) for choosing the majority
    option within each cultural site from a fitted statsmodels Logit result.
    
    Returns a dictionary with:
      - "object": dict mapping each culture level to estimated slope (log-odds per year),
                  standard error, z, two-sided p-value, and 95% CI.
      - "description": brief explanation of what the numbers mean.
    """
    import re
    import numpy as np
    from math import sqrt
    try:
        from scipy import stats
    except Exception:
        # fallback to approximate normal cdf if scipy not available
        def _norm_cdf(x):
            # Abramowitz & Stegun approximation via erf
            import math
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        stats = type("s", (), {"norm": type("n", (), {"cdf": staticmethod(_norm_cdf)})})
    
    # Ensure object looks like a statsmodels results object
    params = getattr(model_output, "params", None)
    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        pass
    if params is None or cov is None:
        raise ValueError("model_output does not appear to be a fitted statsmodels results object with params and cov_params().")
    
    # Must have age_c main effect
    if 'age_c' not in params.index:
        raise ValueError("Fitted model does not contain parameter 'age_c'.")
    
    # Try to get the observed culture levels from the original data if available
    observed_levels = None
    try:
        df = model_output.model.data.frame
        if 'culture' in df.columns:
            # preserve observed order of appearance
            observed_levels = list(dict.fromkeys(df['culture'].tolist()))
    except Exception:
        observed_levels = None
    
    # Find which culture levels appear as main-effect parameters (these are non-reference levels)
    main_culture_levels = []
    for name in params.index:
        m = re.search(r'C\(culture\)\[T\.(.*?)\]', name)
        if m:
            main_culture_levels.append(m.group(1))
    main_culture_levels = list(dict.fromkeys(main_culture_levels))  # unique preserve order
    
    # Determine reference culture:
    reference = None
    if observed_levels:
        # reference is the one in observed_levels not present in main_culture_levels
        for lvl in observed_levels:
            if lvl not in main_culture_levels:
                reference = lvl
                break
        if reference is None:
            # fallback: choose first observed
            reference = observed_levels[0]
    else:
        # fallback: try to infer from parameter names; if we have at least one main level,
        # pick a placeholder reference name
        if main_culture_levels:
            reference = "(reference, not explicitly named in params)"
        else:
            # only one culture present in data
            reference = "(only one culture)"
    
    # Prepare output structure
    slopes = {}
    base_coef = float(params['age_c'])
    var_age = float(cov.loc['age_c', 'age_c'])
    
    # Helper to build interaction param name for a given level
    def interaction_param_name(lvl):
        return f'age_c:C(culture)[T.{lvl}]'
    
    # Iterate over culture levels to compute slope (age effect) per culture
    cultures_to_report = []
    if observed_levels:
        cultures_to_report = observed_levels
    else:
        # construct list from main effects + reference placeholder
        cultures_to_report = [reference] + main_culture_levels
    
    for lvl in cultures_to_report:
        if lvl == reference:
            coef = base_coef
            se = sqrt(var_age)
            inter_name = None
        else:
            inter_name = None
            # find exact interaction parameter name key in params.index (robust to ordering)
            pattern = rf'age_c:C\(culture\)\[T\.{re.escape(str(lvl))}\]'
            for name in params.index:
                if re.fullmatch(pattern, name):
                    inter_name = name
                    break
            if inter_name is None:
                # try other possible ordering (C(culture)[T.X]:age_c)
                pattern2 = rf'C\(culture\)\[T\.{re.escape(str(lvl))}\]:age_c'
                for name in params.index:
                    if re.fullmatch(pattern2, name):
                        inter_name = name
                        break
            if inter_name is None:
                # If no explicit interaction parameter, assume zero interaction (i.e., same slope as reference)
                inter_coef = 0.0
                inter_var = 0.0
                cov_age_inter = 0.0
            else:
                inter_coef = float(params[inter_name])
                inter_var = float(cov.loc[inter_name, inter_name])
                cov_age_inter = float(cov.loc['age_c', inter_name])
            coef = base_coef + (inter_coef if inter_name is not None else 0.0)
            # compute variance of sum
            se = sqrt(var_age + (inter_var if inter_name is not None else 0.0) + 2.0 * (cov_age_inter if inter_name is not None else 0.0))
        
        z = coef / se if se > 0 else np.nan
        p_two = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        
        slopes[str(lvl)] = {
            "slope_log_odds_per_year": coef,
            "se": se,
            "z": z,
            "p_two_sided": p_two,
            "95ci": (ci_lower, ci_upper),
            "note": "Slope is the change in log-odds of choosing the majority per 1 year increase in age in this culture."
        }
    
    description = (
        "For each cultural site, the function returns the estimated age slope (in log-odds per year),\n"
        "its standard error, z-statistic, two-sided p-value testing slope != 0, and a 95% CI.\n"
        "Positive slope => increasing probability of choosing the majority with age; negative => decreasing.\n"
        f"The reference culture (baseline in the model) is: {reference}.\n"
        "If an interaction term for a culture is present, that culture's slope = (age_c main effect) + (age_c:C(culture)[T.<level>] interaction).\n"
        "Standard errors for non-reference slopes account for covariances between the main age effect and the interaction term."
    )
    
    return {"object": slopes, "description": description}