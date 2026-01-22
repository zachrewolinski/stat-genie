def extract_final_answer(model_output):
    """
    Extract coefficients, SEs, z-stats, p-values, 95% CIs, and odds-ratios for the
    focal predictors from a fitted statsmodels GLMResults/GLMResultsWrapper object.

    Returns a dictionary with:
      - "object": dict mapping variable -> extracted numeric summary
      - "description": human-readable interpretation focusing on the key predictors
    """
    import numpy as np
    import math

    # Variables of primary interest
    focal_vars = ['log_size_ratio_c', 'focal_dist_m_c', 'size_x_focaldist']

    # Prepare outputs
    stats = {}
    missing_vars = []

    # Try to access usual attributes on statsmodels result object
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf = model_output.conf_int()
    except Exception as e:
        raise ValueError("Provided model_output does not have expected statsmodels attributes: " + str(e))

    # Helper to safely get numeric value for a variable (if present)
    def get_val(series_or_df, var, default=math.nan):
        try:
            return float(series_or_df[var])
        except Exception:
            # conf_int returns DataFrame with two columns (0,1) — handle that
            try:
                row = series_or_df.loc[var]
                # If it's a length-2 array-like, return as tuple
                if hasattr(row, "__len__") and len(row) >= 2:
                    return (float(row[0]), float(row[1]))
            except Exception:
                return default

    # Extract stats for each focal var
    for v in focal_vars:
        if v not in params.index:
            missing_vars.append(v)
            continue
        coef = get_val(params, v)
        se = get_val(bse, v)
        p = get_val(pvalues, v)
        # z-stat
        z = coef / se if (not math.isnan(coef) and not math.isnan(se) and se != 0) else math.nan
        # 95% CI on coefficient scale
        ci = None
        try:
            ci_row = conf.loc[v]
            ci = (float(ci_row[0]), float(ci_row[1]))
        except Exception:
            ci = (math.nan, math.nan)
        # Odds ratio and CI
        or_val = float(np.exp(coef)) if not math.isnan(coef) else math.nan
        or_ci = (float(np.exp(ci[0])) if not math.isnan(ci[0]) else math.nan,
                 float(np.exp(ci[1])) if not math.isnan(ci[1]) else math.nan)

        stats[v] = {
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'ci_2.5%': ci[0],
            'ci_97.5%': ci[1],
            'odds_ratio': or_val,
            'odds_ratio_ci_2.5%': or_ci[0],
            'odds_ratio_ci_97.5%': or_ci[1],
            'significant_0.05': bool(p < 0.05)
        }

    # Also include basic model-level info if available (N, llf)
    model_info = {}
    try:
        model_info['n_obs'] = int(model_output.nobs)
    except Exception:
        model_info['n_obs'] = None
    try:
        model_info['deviance'] = float(model_output.deviance)
    except Exception:
        model_info['deviance'] = None

    # Build a concise human-readable description using the extracted numbers
    lines = []
    if missing_vars:
        lines.append("Warning: the following focal variables were not found in the model output: " +
                     ", ".join(missing_vars) + ".")
    lines.append(f"Model sample size (n_obs): {model_info.get('n_obs')}")
    for v in focal_vars:
        if v not in stats:
            continue
        s = stats[v]
        sig = "statistically significant (p < 0.05)" if s['significant_0.05'] else "not statistically significant (p ≥ 0.05)"
        # Interpret sign for direction
        direction = "positive" if s['coef'] > 0 else ("negative" if s['coef'] < 0 else "near zero")
        # Special interpretation for focal_dist_m_c: increasing distance = farther from home
        if v == 'focal_dist_m_c':
            meaning = ("This coefficient is per meter increase in focal distance from home; "
                       "a negative coef means being closer to home (smaller distance) increases the focal group's odds of winning.")
        elif v == 'log_size_ratio_c':
            meaning = ("This coefficient is per unit increase in the log ratio of focal:other adult group size; "
                       "a positive coef means a larger focal group (relative to the opponent) increases the odds the focal group wins.")
        elif v == 'size_x_focaldist':
            meaning = ("This is the interaction term between relative size and focal distance; "
                       "a significant interaction means the effect of relative size on winning depends on contest location (distance).")
        else:
            meaning = ""
        line = (f"{v}: coef={s['coef']:.4f}, SE={s['se']:.4f}, z={s['z']:.2f}, p={s['p']:.3g}; "
                f"OR={s['odds_ratio']:.3f} (95% CI [{s['odds_ratio_ci_2.5%']:.3f}, {s['odds_ratio_ci_97.5%']:.3f}]); "
                f"Direction: {direction}; {sig}. {meaning}")
        lines.append(line)

    description = " ".join(lines)

    return {"object": stats, "description": description}