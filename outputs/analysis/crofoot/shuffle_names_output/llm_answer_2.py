def extract_final_answer(model_output):
    """
    Extracts and interprets statistics from a fitted statsmodels GLMResults-like object
    (ideally the cluster-robust result returned by get_robustcov_results).
    Returns a dictionary with:
      - "object": dict of extracted effects (estimates, SE, z, p, 95% CI, odds ratio and OR CI)
      - "description": brief explanation of what the results mean
    
    Expected coefficient names in the model:
      'log_rel_size', 'log_size_x_Focal', 'log_size_x_Other',
      'ContestLoc_Focal', 'ContestLoc_Other'
    """
    import numpy as np
    from scipy.stats import norm
    from collections import OrderedDict

    res = model_output

    # Helper: get params series and covariance matrix as DataFrame-like (indexable by names)
    try:
        params = res.params.copy()
    except Exception as e:
        raise ValueError(f"Could not obtain params from model_output: {e}")

    try:
        cov = res.cov_params()
    except Exception:
        # fallback to using outer product of bse if cov not available (less preferred)
        try:
            bse = res.bse
            idx = params.index
            cov = np.diag(bse**2)
            # convert to DataFrame-like with index for easier access
            import pandas as _pd
            cov = _pd.DataFrame(cov, index=idx, columns=idx)
        except Exception as e:
            raise ValueError(f"Could not obtain covariance matrix (or bse) from model_output: {e}")

    # If cov is numpy array, convert to DataFrame using params.index
    try:
        import pandas as pd
        if isinstance(cov, np.ndarray):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        pass

    # Names we'll extract / combine
    coeff_names = [
        'log_rel_size',
        'log_size_x_Focal',
        'log_size_x_Other',
        'ContestLoc_Focal',
        'ContestLoc_Other'
    ]

    # Validate presence of required coefficient names
    missing = [n for n in coeff_names if n not in params.index]
    if missing:
        # If interaction terms absent, still proceed with what exists
        # but inform the user
        missing_msg = f"Warning: the following expected coefficients are missing from the model: {missing}"
    else:
        missing_msg = None

    # Function to compute linear combination stats given a dict of {name: weight}
    def lincomb_stats(weights):
        # weights: dict mapping coef name -> multiplier
        # compute estimate
        est = 0.0
        for name, w in weights.items():
            if name not in params.index:
                raise KeyError(f"Coefficient '{name}' not present in model params.")
            est += w * params[name]
        # variance
        var = 0.0
        for i, (n1, w1) in enumerate(weights.items()):
            for j, (n2, w2) in enumerate(weights.items()):
                var += w1 * w2 * cov.loc[n1, n2]
        se = np.sqrt(var) if var >= 0 else np.nan
        z = est / se if se and not np.isnan(se) else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = est - norm.ppf(0.975) * se if not np.isnan(se) else np.nan
        ci_high = est + norm.ppf(0.975) * se if not np.isnan(se) else np.nan
        # odds ratio and CI
        or_est = np.exp(est) if not np.isnan(est) else np.nan
        or_ci = (np.exp(ci_low), np.exp(ci_high)) if not np.isnan(ci_low) else (np.nan, np.nan)
        return {
            'estimate_log_odds': float(est),
            'se': float(se),
            'z': float(z),
            'p_value': float(p),
            'ci_95_log_odds': (float(ci_low), float(ci_high)),
            'odds_ratio': float(or_est),
            'ci_95_odds_ratio': (float(or_ci[0]), float(or_ci[1])),
            'weights': dict(weights)
        }

    results = OrderedDict()

    # 1) Main effect of log_rel_size (this is the effect when both location dummies = 0,
    # i.e., Neutral contests)
    if 'log_rel_size' in params.index:
        try:
            results['log_rel_size_neutral'] = lincomb_stats({'log_rel_size': 1.0})
        except Exception as e:
            results['log_rel_size_neutral'] = {'error': str(e)}
    else:
        results['log_rel_size_neutral'] = {'error': "log_rel_size coefficient not present"}

    # 2) Effect of log_rel_size when contest is at Focal location:
    #    log_rel_size + log_size_x_Focal
    if 'log_rel_size' in params.index and 'log_size_x_Focal' in params.index:
        try:
            results['log_rel_size_at_Focal'] = lincomb_stats({'log_rel_size': 1.0, 'log_size_x_Focal': 1.0})
        except Exception as e:
            results['log_rel_size_at_Focal'] = {'error': str(e)}
    else:
        results['log_rel_size_at_Focal'] = {'error': "Required coefficients for Focal interaction not present"}

    # 3) Effect of log_rel_size when contest is at Other location:
    #    log_rel_size + log_size_x_Other
    if 'log_rel_size' in params.index and 'log_size_x_Other' in params.index:
        try:
            results['log_rel_size_at_Other'] = lincomb_stats({'log_rel_size': 1.0, 'log_size_x_Other': 1.0})
        except Exception as e:
            results['log_rel_size_at_Other'] = {'error': str(e)}
    else:
        results['log_rel_size_at_Other'] = {'error': "Required coefficients for Other interaction not present"}

    # 4) Main effects of Contest location dummies (these are differences in intercept by location,
    #    holding other covariates at reference). Extract raw coef stats if present.
    for loc in ['ContestLoc_Focal', 'ContestLoc_Other']:
        if loc in params.index:
            est = float(params[loc])
            se = float(cov.loc[loc, loc]**0.5) if loc in cov.index else float(np.nan)
            z = est / se if se and not np.isnan(se) else float(np.nan)
            p = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else float(np.nan)
            ci_low = est - norm.ppf(0.975) * se if not np.isnan(se) else float(np.nan)
            ci_high = est + norm.ppf(0.975) * se if not np.isnan(se) else float(np.nan)
            results[loc] = {
                'estimate_log_odds': est,
                'se': se,
                'z': z,
                'p_value': p,
                'ci_95_log_odds': (ci_low, ci_high),
                'odds_ratio': float(np.exp(est)),
                'ci_95_odds_ratio': (float(np.exp(ci_low)), float(np.exp(ci_high)))
            }
        else:
            results[loc] = {'error': f"{loc} not present in model parameters"}

    # 5) Also include the raw parameter table (est, se, p, 95% CI) for transparency if available
    try:
        # Some result objects include .pvalues and .bse and .conf_int()
        param_table = {}
        for name in params.index:
            try:
                est = float(params[name])
                se = float(res.bse[name]) if hasattr(res, 'bse') else float(np.sqrt(cov.loc[name, name]))
                z = est / se if se and not np.isnan(se) else float(np.nan)
                p = float(res.pvalues[name]) if hasattr(res, 'pvalues') else float(2 * (1 - norm.cdf(abs(z))))
                ci_low, ci_high = (None, None)
                try:
                    ci = res.conf_int().loc[name]
                    ci_low, ci_high = float(ci[0]), float(ci[1])
                except Exception:
                    ci_low = est - norm.ppf(0.975) * se
                    ci_high = est + norm.ppf(0.975) * se
                param_table[name] = {
                    'estimate_log_odds': est,
                    'se': se,
                    'z': z,
                    'p_value': p,
                    'ci_95_log_odds': (ci_low, ci_high),
                    'odds_ratio': float(np.exp(est)),
                    'ci_95_odds_ratio': (float(np.exp(ci_low)), float(np.exp(ci_high)))
                }
            except Exception:
                param_table[name] = {'error': 'could not extract'}
        results['parameter_table'] = param_table
    except Exception:
        results['parameter_table'] = {'error': 'could not build parameter table'}

    # Build a short description of what the returned object contains and how to interpret
    description_lines = [
        "Returned entries give estimated effects on the log-odds (and transformed odds ratios) that the focal group wins.",
        "- 'log_rel_size_neutral': effect of log(total_focal/total_other) when contest location is Neutral (both location dummies = 0).",
        "- 'log_rel_size_at_Focal': combined effect (main + interaction) when contest occurs nearer the focal group's home-range center.",
        "- 'log_rel_size_at_Other': combined effect (main + interaction) when contest occurs nearer the other group's home-range center.",
        "- 'ContestLoc_Focal' and 'ContestLoc_Other': estimated shift in intercept (log-odds) for contests at those locations relative to the baseline (Neutral).",
        "- Each effect contains: estimate (log-odds), SE, z-stat, two-sided p-value, 95% CI on log-odds, odds ratio and 95% CI on odds ratio.",
    ]
    if missing_msg:
        description_lines.append(missing_msg)

    description = " ".join(description_lines)

    return {"object": results, "description": description}