def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for:
      - the main effect of Beauty
      - the Beauty x Female interaction
      - simple slopes of Beauty for males (Female=0) and females (Female=1)

    Returns a dict with keys:
      - "object": a dict of numeric results (all Python floats)
      - "description": brief interpretation of the extracted numbers

    Expects a statsmodels RegressionResults-like object (RegressionResultsWrapper).
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Get parameter names and values
    params = res.params
    param_names = list(params.index)

    # Helper to find exact or interaction parameter names robustly
    def find_param(name):
        if name in param_names:
            return name
        # allow for possible naming variations (e.g., interaction order)
        matches = [n for n in param_names if n == name]
        return matches[0] if matches else None

    def find_interaction(a, b):
        # find a parameter name that contains both tokens a and b separated by ':'
        for n in param_names:
            if ':' in n:
                parts = n.split(':')
                if a in parts and b in parts:
                    return n
        return None

    # Identify parameter names
    beauty_name = find_param('Beauty')
    if beauty_name is None:
        raise ValueError("Could not find parameter named 'Beauty' in model parameters.")

    interaction_name = find_interaction('Beauty', 'Female')
    if interaction_name is None:
        # Try alternative token order or underscore variants
        interaction_name = find_interaction('Female', 'Beauty')

    # Build covariance matrix for linear combinations (should reflect cluster-robust cov if model was fitted that way)
    cov = res.cov_params()

    # Extract main effect stats directly if present
    def extract_param_stats(name):
        est = float(params[name])
        # If cov has matching index names
        se = float(np.sqrt(float(cov.loc[name, name])))
        # t-stat from estimate / se
        t_stat = est / se if se != 0 else np.nan
        # df for t distribution (use df_resid if available)
        df = getattr(res, 'df_resid', None)
        if df is not None and np.isfinite(df) and df > 0:
            pval = float(2.0 * stats.t.sf(np.abs(t_stat), df))
            crit = stats.t.ppf(0.975, df)
        else:
            pval = float(2.0 * stats.norm.sf(np.abs(t_stat)))
            crit = stats.norm.ppf(0.975)
        ci_low = est - crit * se
        ci_high = est + crit * se
        return {
            'estimate': est,
            'se': se,
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_low),
            'ci_upper': float(ci_high),
        }

    main_beauty_stats = extract_param_stats(beauty_name)

    interaction_stats = None
    if interaction_name is not None:
        interaction_stats = extract_param_stats(interaction_name)
    else:
        # If no interaction term found, set to None
        interaction_stats = None

    # Compute simple slopes for males (Female=0) and females (Female=1)
    # Build linear combination vectors L such that slope = L' * params
    # Ensure cov and params have same ordering; create numpy arrays in that order.
    param_order = list(param_names)
    param_vector = np.array([float(params[n]) for n in param_order])
    cov_matrix = np.array(cov.loc[param_order, param_order], dtype=float)

    def linear_combination(L):
        est = float(np.dot(L, param_vector))
        var = float(np.dot(L, np.dot(cov_matrix, L)))
        se = float(np.sqrt(var)) if var >= 0 else float(np.nan)
        t_stat = est / se if se != 0 else float('nan')
        df = getattr(res, 'df_resid', None)
        if df is not None and np.isfinite(df) and df > 0:
            pval = float(2.0 * stats.t.sf(np.abs(t_stat), df))
            crit = stats.t.ppf(0.975, df)
        else:
            pval = float(2.0 * stats.norm.sf(np.abs(t_stat)))
            crit = stats.norm.ppf(0.975)
        ci_low = est - crit * se
        ci_high = est + crit * se
        return {
            'estimate': est,
            'se': se,
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_low),
            'ci_upper': float(ci_high),
        }

    # L for males: selects Beauty coefficient only (Female=0, so interaction not included)
    L_male = np.zeros(len(param_order))
    L_male[param_order.index(beauty_name)] = 1.0

    male_slope_stats = linear_combination(L_male)

    # L for females: Beauty + (Beauty:Female) if interaction exists
    L_female = np.zeros(len(param_order))
    L_female[param_order.index(beauty_name)] = 1.0
    if interaction_name is not None:
        L_female[param_order.index(interaction_name)] = 1.0

    female_slope_stats = linear_combination(L_female)

    # Prepare return object with python-native floats
    result_object = {
        'beauty_param_name': beauty_name,
        'beauty': main_beauty_stats,
        'interaction_param_name': interaction_name,
        'interaction': interaction_stats,
        'simple_slope_male (Female=0)': male_slope_stats,
        'simple_slope_female (Female=1)': female_slope_stats,
        'notes': (
            "Estimates show change in Eval (1-5 scale) per one-unit increase in mean-centered Beauty. "
            "Simple slopes give the effect separately for male instructors (Female=0) and female instructors (Female=1). "
            "P-values are two-sided; 95% CIs shown."
        )
    }

    description = (
        "This output returns the estimated coefficient, standard error, t-statistic, two-sided p-value, "
        "and 95% confidence interval for the main effect of instructor beauty (Beauty), the Beauty x Female "
        "interaction (if present), and the estimated simple slopes of Beauty for males and females. "
        "Positive estimates indicate higher teaching evaluations for higher beauty scores; p-values indicate "
        "whether those estimates differ from zero."
    )

    return {"object": result_object, "description": description}