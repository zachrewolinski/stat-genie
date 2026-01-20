def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear slope per cultural site and global quadratic term)
    from a fitted statsmodels Logit (BinaryResultsWrapper) object.

    Returns a dictionary with:
      - "object": a structured dict containing:
          - per_culture: for each culture (including the omitted/reference category),
              - slope_logodds: linear age coefficient on log-odds scale (age_c effect for that culture)
              - se: standard error of that slope (Wald se using covariance matrix)
              - z: z-statistic
              - p: two-sided p-value
              - ci95: (lower, upper) 95% CI on log-odds
              - odds_ratio: exp(slope_logodds)
              - or_ci95: (lower, upper) 95% CI for odds ratio
          - age_c2: dict for the global quadratic term (coef, se, z, p, ci95)
      - "description": short plain-language interpretation of what the numbers mean.
    """
    import numpy as np
    from math import exp, sqrt
    try:
        # scipy may not always be available in every runtime; use it if present for p-values
        from scipy.stats import norm
        _norm_sf = lambda x: norm.sf(x)
    except Exception:
        # fallback: approximate two-sided p using normal CDF from numpy (erf)
        import math
        def _norm_sf(x):
            # survival function for standard normal using erf
            return 0.5 * (1.0 - math.erf(x / math.sqrt(2.0)))

    res = model_output  # statsmodels BinaryResultsWrapper

    params = res.params
    cov = res.cov_params()

    # ensure required age term exists
    if 'age_c' not in params.index:
        raise KeyError("Model output does not contain 'age_c' coefficient; cannot extract age effects.")

    beta_age = float(params['age_c'])
    var_age = float(cov.loc['age_c', 'age_c']) if ('age_c' in cov.index and 'age_c' in cov.columns) else None

    results_per_culture = {}

    # identify culture dummy names: those starting with 'culture_' but NOT containing '_x_age'
    culture_dummies = [name for name in params.index if name.startswith('culture_') and '_x_age' not in name]

    # The omitted/reference culture is the one that was dropped in get_dummies (no dummy). We'll label it "reference (omitted)".
    # Compute slope for the reference (omitted) culture:
    if var_age is None:
        raise ValueError("Covariance for 'age_c' not found in model covariance matrix.")
    se_age = sqrt(var_age)
    z_age = beta_age / se_age if se_age > 0 else float('nan')
    p_age = float(2.0 * _norm_sf(abs(z_age)))
    ci_lower = beta_age - 1.96 * se_age
    ci_upper = beta_age + 1.96 * se_age
    results_per_culture['reference (omitted)'] = {
        'slope_logodds': float(beta_age),
        'se': float(se_age),
        'z': float(z_age),
        'p': float(p_age),
        'ci95': (float(ci_lower), float(ci_upper)),
        'odds_ratio': float(exp(beta_age)),
        'or_ci95': (float(exp(ci_lower)), float(exp(ci_upper)))
    }

    # For each explicit culture dummy, compute slope = beta_age + beta_interaction
    for dummy in culture_dummies:
        interaction_name = f"{dummy}_x_age"
        beta_inter = float(params[interaction_name]) if interaction_name in params.index else 0.0

        # variance: var(age) + var(inter) + 2*cov(age, inter)
        var_inter = float(cov.loc[interaction_name, interaction_name]) if (interaction_name in cov.index and interaction_name in cov.columns) else 0.0
        cov_age_inter = float(cov.loc['age_c', interaction_name]) if ('age_c' in cov.index and interaction_name in cov.columns) else 0.0

        slope = beta_age + beta_inter
        var_slope = var_age + var_inter + 2.0 * cov_age_inter
        se_slope = sqrt(var_slope) if var_slope >= 0 else float('nan')
        z_slope = slope / se_slope if se_slope > 0 else float('nan')
        p_slope = float(2.0 * _norm_sf(abs(z_slope))) if not np.isnan(z_slope) else float('nan')
        ci_low = slope - 1.96 * se_slope
        ci_high = slope + 1.96 * se_slope

        # derive culture label by removing 'culture_' prefix (if any)
        label = dummy[len('culture_'):] if dummy.startswith('culture_') else dummy

        results_per_culture[label] = {
            'slope_logodds': float(slope),
            'se': float(se_slope),
            'z': float(z_slope),
            'p': float(p_slope),
            'ci95': (float(ci_low), float(ci_high)),
            'odds_ratio': float(exp(slope)),
            'or_ci95': (float(exp(ci_low)), float(exp(ci_high)))
        }

    # Global quadratic age term (age_c2) - same across cultures in the fitted model
    age2_info = None
    if 'age_c2' in params.index:
        beta_age2 = float(params['age_c2'])
        var_age2 = float(cov.loc['age_c2', 'age_c2']) if ('age_c2' in cov.index and 'age_c2' in cov.columns) else None
        if var_age2 is None:
            se_age2 = float('nan')
            z_age2 = float('nan')
            p_age2 = float('nan')
            ci_age2 = (float('nan'), float('nan'))
        else:
            se_age2 = sqrt(var_age2)
            z_age2 = beta_age2 / se_age2 if se_age2 > 0 else float('nan')
            p_age2 = float(2.0 * _norm_sf(abs(z_age2)))
            ci_age2 = (float(beta_age2 - 1.96 * se_age2), float(beta_age2 + 1.96 * se_age2))
        age2_info = {
            'coef': float(beta_age2),
            'se': float(se_age2),
            'z': float(z_age2),
            'p': float(p_age2),
            'ci95': ci_age2,
            'note': "This quadratic term applies globally (not interacted with culture) and captures curvature in the age effect on log-odds."
        }

    output_object = {
        'per_culture': results_per_culture,
        'age_c2': age2_info
    }

    description = (
        "For each culture this returns the linear age slope on the log-odds scale for choosing the majority (slope_logodds),\n"
        "its standard error, Wald z-statistic, two-sided p-value, 95% CI, and the corresponding odds ratio (with CI).\n"
        "The 'reference (omitted)' entry is the baseline culture that was omitted when creating dummy variables; "
        "explicit culture labels correspond to dummy variables in the model.\n"
        "A positive slope_logodds indicates increasing reliance on the majority with age (on the log-odds scale); "
        "if the p-value is small (e.g., < .05) this change is statistically significant. The global quadratic term (age_c2)\n"
        "describes curvature (acceleration/deceleration) of the age trajectory and applies to all cultures because it\n"
        "was not interacted with culture in the model."
    )

    return {"object": output_object, "description": description}