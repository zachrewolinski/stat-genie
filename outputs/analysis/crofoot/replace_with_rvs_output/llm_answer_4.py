def extract_final_answer(model_output):
    """
    Extract coefficients, clustered-robust standard errors, z-stats, p-values,
    95% CIs, and odds-ratios for the key predictors:
      - RelSize_z
      - RelDist_z
      - RelSize_x_RelDist

    Returns a dict with:
      - "object": dict of numeric results for each predictor
      - "description": concise interpretation of those results in the study context
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Parameter estimates (expecting a pandas Series)
    params = res.params.copy()

    # Use the returned covariance matrix (should reflect clustered cov if fit used cov_type='cluster')
    cov = res.cov_params()

    # Ensure covariance is a numpy array for diag; handle DataFrame or ndarray
    cov_mat = np.asarray(cov)

    # Standard errors from covariance matrix
    se_arr = np.sqrt(np.diag(cov_mat))
    se = pd.Series(se_arr, index=params.index)

    # z-statistics (Wald z using Normal approx)
    z_stats = params / se

    # two-sided p-values (ensure p_values is a Series with same index)
    p_values = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z_stats))), index=z_stats.index)

    # 95% confidence intervals on the log-odds scale using standard normal critical value
    z_crit = stats.norm.ppf(0.975)
    ci_lower = params - z_crit * se
    ci_upper = params + z_crit * se

    # Prepare outputs for the focal predictors
    focal_predictors = ['RelSize_z', 'RelDist_z', 'RelSize_x_RelDist']
    out = {}
    for pred in focal_predictors:
        if pred in params.index:
            coef = float(params[pred])
            se_val = float(se[pred])
            z_val = float(z_stats[pred])
            p_val = float(p_values[pred])
            ci_l = float(ci_lower[pred])
            ci_u = float(ci_upper[pred])
            odds_ratio = float(np.exp(coef))
            or_ci = (float(np.exp(ci_l)), float(np.exp(ci_u)))

            out[pred] = {
                'coef_log_odds': coef,
                'se': se_val,
                'z': z_val,
                'p_value': p_val,
                '95%_CI_log_odds': (ci_l, ci_u),
                'odds_ratio': odds_ratio,
                '95%_CI_odds_ratio': or_ci
            }
        else:
            out[pred] = None

    # Build a concise interpretation string
    interp_lines = []
    for pred in focal_predictors:
        info = out[pred]
        if info is None:
            interp_lines.append(f"{pred}: not present in model output.")
            continue
        sig = info['p_value'] < 0.05
        direction = 'positive' if info['coef_log_odds'] > 0 else ('negative' if info['coef_log_odds'] < 0 else 'null')
        interp = (f"{pred}: coef={info['coef_log_odds']:.3f}, p={info['p_value']:.3f} "
                  f"({'significant' if sig else 'ns'}). Direction={direction}. "
                  f"OR={info['odds_ratio']:.3f}, 95% CI OR=({info['95%_CI_odds_ratio'][0]:.3f}, {info['95%_CI_odds_ratio'][1]:.3f}).")
        interp_lines.append(interp)

    # Brief contextual summary
    summary = (
        "Interpretation: Positive coefficients indicate higher log-odds (and OR>1) of the focal group winning. "
        "RelSize_z tests whether being larger than the opponent increases win probability. "
        "RelDist_z tests whether being closer to the focal home-range center (i.e., contest in focal home range) increases win probability. "
        "RelSize_x_RelDist tests whether the effect of relative size depends on contest location (a significant interaction means the size advantage differs depending on location)."
    )

    description = "\n".join(interp_lines) + "\n\n" + summary

    return {"object": out, "description": description}