def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM model (with optional
    cluster-robust results) and produce an interpretable summary focused on:
      - z_RelativeSize (main effect)
      - z_LocAdv (main effect)
      - interaction z_RelativeSize:z_LocAdv

    Returns a dict with:
      - "object": a dictionary containing tables of coefficients (coef, se, p,
                  95% CI) and odds-ratios, plus marginal effects of a one-SD
                  increase in relative size at LocAdv = -1, 0, +1.
      - "description": a short plain-language interpretation of what the
                       extracted statistics mean for the task.

    model_output is expected to be the dict returned by the supplied model()
    function, typically containing at least the key 'fit' and possibly a
    cluster-robust results object under 'cluster_robust'.
    """
    import numpy as np
    import pandas as pd
    from math import sqrt
    from scipy import stats

    def safe_exp(x):
        """
        Compute exponential in a way that avoids raising OverflowError.
        Uses numpy.exp and maps extreme values to inf/0 as appropriate.
        """
        try:
            val = np.exp(x)
            # numpy.exp returns numpy types; convert to Python float if finite
            if np.isscalar(val):
                if np.isfinite(val):
                    return float(val)
                # val is inf or nan
                if np.isnan(val):
                    # treat nan conservatively as inf for positive x, 0 for negative x
                    return float('inf') if x >= 0 else 0.0
                return float('inf') if val > 0 else 0.0
            else:
                # fallback: convert elementwise if needed
                val = np.array(val, dtype=float)
                if np.all(np.isfinite(val)):
                    return float(val)
                return float('inf') if x >= 0 else 0.0
        except Exception:
            # If anything unexpected happens, fall back to safe heuristic
            try:
                xv = float(x)
            except Exception:
                return float('inf')
            return float('inf') if xv >= 0 else 0.0

    # Validate and select results
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model() function.")
    results = model_output.get('cluster_robust', model_output.get('fit'))
    if results is None:
        raise ValueError("model_output does not contain 'fit' or 'cluster_robust' results.")

    # Extract parameter info
    params = getattr(results, 'params', None)
    bse = getattr(results, 'bse', None)
    pvalues = getattr(results, 'pvalues', None)

    # Confidence intervals (try standard method, else approximate)
    try:
        ci_df = results.conf_int()
    except Exception:
        if bse is None or params is None:
            raise
        ci_df = pd.DataFrame({
            0: params - 1.96 * bse,
            1: params + 1.96 * bse
        }, index=params.index)

    # Covariance matrix if available
    cov = None
    try:
        cov = results.cov_params()
    except Exception:
        cov = None

    # Terms of interest
    terms = ['z_RelativeSize', 'z_LocAdv', 'z_RelativeSize:z_LocAdv']
    term_table = {}
    for t in terms:
        if params is not None and t in params.index:
            coef = float(params.loc[t])
            se = float(bse.loc[t]) if bse is not None and t in getattr(bse, 'index', []) else None
            p = float(pvalues.loc[t]) if pvalues is not None and t in getattr(pvalues, 'index', []) else None
            # CI might have different column labels; try common ones
            try:
                ci_low = float(ci_df.loc[t, 0])
                ci_high = float(ci_df.loc[t, 1])
            except Exception:
                # attempt named columns
                try:
                    ci_low = float(ci_df.loc[t, ci_df.columns[0]])
                    ci_high = float(ci_df.loc[t, ci_df.columns[1]])
                except Exception:
                    ci_low = None
                    ci_high = None

            # Odds ratio and CI on OR scale using safe_exp
            or_val = safe_exp(coef)
            or_ci_low = safe_exp(ci_low) if ci_low is not None else None
            or_ci_high = safe_exp(ci_high) if ci_high is not None else None

            term_table[t] = {
                'coef_log_odds': coef,
                'se': se,
                'p_value': p,
                '95CI_log_odds': (ci_low, ci_high) if (ci_low is not None and ci_high is not None) else None,
                'OR': or_val,
                '95CI_OR': (or_ci_low, or_ci_high) if (or_ci_low is not None and or_ci_high is not None) else None
            }
        else:
            term_table[t] = None

    # Compute marginal effects of a 1-SD increase in relative size at loc adv -1,0,1
    me_results = {}
    if params is not None and ('z_RelativeSize' in params.index) and ('z_RelativeSize:z_LocAdv' in params.index):
        beta_size = float(params.loc['z_RelativeSize'])
        beta_int = float(params.loc['z_RelativeSize:z_LocAdv'])
        # Attempt to get variances/covariances
        cov_available = False
        if cov is not None:
            try:
                var_size = float(cov.loc['z_RelativeSize', 'z_RelativeSize'])
                var_int = float(cov.loc['z_RelativeSize:z_LocAdv', 'z_RelativeSize:z_LocAdv'])
                cov_size_int = float(cov.loc['z_RelativeSize', 'z_RelativeSize:z_LocAdv'])
                cov_available = True
            except Exception:
                cov_available = False

        for loc in [-1.0, 0.0, 1.0]:
            me_logodds = beta_size + beta_int * loc
            if cov_available:
                var_me = var_size + (loc**2) * var_int + 2 * loc * cov_size_int
                se_me = sqrt(max(var_me, 0.0))
                zstat = me_logodds / se_me if se_me > 0 else float('nan')
                p_me = 2 * (1 - stats.norm.cdf(abs(zstat)))
                ci_low = me_logodds - 1.96 * se_me
                ci_high = me_logodds + 1.96 * se_me
            else:
                se_me = None
                p_me = None
                ci_low = None
                ci_high = None

            me_results[f'LocAdv={loc}'] = {
                'marginal_effect_log_odds_per_1SD_relSize': me_logodds,
                'se': se_me,
                'p_value': p_me,
                '95CI_log_odds': (ci_low, ci_high) if (ci_low is not None and ci_high is not None) else None,
                'OR_for_1SD_in_relSize': safe_exp(me_logodds),
                '95CI_OR': (safe_exp(ci_low), safe_exp(ci_high)) if (ci_low is not None and ci_high is not None) else None
            }
    else:
        # If no interaction, provide overall effect if available
        if params is not None and 'z_RelativeSize' in params.index:
            beta_size = float(params.loc['z_RelativeSize'])
            se = float(bse.loc['z_RelativeSize']) if bse is not None and 'z_RelativeSize' in getattr(bse, 'index', []) else None
            try:
                ci_low = float(ci_df.loc['z_RelativeSize', 0])
                ci_high = float(ci_df.loc['z_RelativeSize', 1])
            except Exception:
                try:
                    ci_low = float(ci_df.loc['z_RelativeSize', ci_df.columns[0]])
                    ci_high = float(ci_df.loc['z_RelativeSize', ci_df.columns[1]])
                except Exception:
                    ci_low = None
                    ci_high = None

            me_results['Overall'] = {
                'marginal_effect_log_odds_per_1SD_relSize': beta_size,
                'se': se,
                'p_value': float(pvalues.loc['z_RelativeSize']) if pvalues is not None and 'z_RelativeSize' in getattr(pvalues, 'index', []) else None,
                '95CI_log_odds': (ci_low, ci_high) if (ci_low is not None and ci_high is not None) else None,
                'OR_for_1SD_in_relSize': safe_exp(beta_size),
                '95CI_OR': (safe_exp(ci_low), safe_exp(ci_high)) if (ci_low is not None and ci_high is not None) else None
            }

    # Interpretation helper
    def interpret_term(tdict):
        if tdict is None:
            return "Term not present in model."
        p = tdict.get('p_value', None)
        coef = tdict.get('coef_log_odds', 0.0)
        orv = tdict.get('OR', None)
        ci_or = tdict.get('95CI_OR', None)
        sig_text = ""
        if p is not None:
            if p < 0.001:
                sig_text = "strong evidence (p < 0.001)"
            elif p < 0.01:
                sig_text = "strong evidence (p < 0.01)"
            elif p < 0.05:
                sig_text = "moderate evidence (p < 0.05)"
            else:
                sig_text = f"no strong evidence (p = {p:.3f})"
        else:
            sig_text = "p-value not available"

        direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")

        # Prepare OR and CI text robustly
        try:
            or_text = f"{orv:.3f}" if orv is not None else "N/A"
        except Exception:
            or_text = str(orv)
        if ci_or is not None and len(ci_or) == 2 and ci_or[0] is not None and ci_or[1] is not None:
            try:
                ci_text = f"{ci_or[0]:.3f}–{ci_or[1]:.3f}"
            except Exception:
                ci_text = f"{ci_or[0]}–{ci_or[1]}"
        else:
            ci_text = "N/A"

        return (f"Coef (log-odds) = {coef:.3f}; OR = {or_text} (95% CI {ci_text}); "
                f"{direction} in probability of focal-group win per 1 SD increase. {sig_text}.")

    interp_lines = []
    interp_lines.append("Key predictors and their interpretation:")
    interp_lines.append("z_RelativeSize: " + interpret_term(term_table.get('z_RelativeSize')))
    interp_lines.append("z_LocAdv (home advantage): " + interpret_term(term_table.get('z_LocAdv')))
    interp_lines.append("Interaction (z_RelativeSize:z_LocAdv): " + interpret_term(term_table.get('z_RelativeSize:z_LocAdv')))
    interp_lines.append("")
    interp_lines.append("Marginal effects of a 1-SD increase in relative group size at example location-advantage values:")
    for k, v in me_results.items():
        try:
            me_log = v['marginal_effect_log_odds_per_1SD_relSize']
            or_me = v['OR_for_1SD_in_relSize']
            me_desc = f"{k}: log-odds change = {me_log:.3f}; OR = {or_me:.3f}"
        except Exception:
            me_desc = f"{k}: log-odds change = {v.get('marginal_effect_log_odds_per_1SD_relSize')}; OR = {v.get('OR_for_1SD_in_relSize')}"
        if v.get('p_value') is not None:
            me_desc += f"; p = {v['p_value']:.3f}"
        interp_lines.append(me_desc)

    description = "\n".join(interp_lines)

    summary_object = {
        'term_table': term_table,
        'marginal_effects_of_relSize_at_LocAdv': me_results,
        'notes': ("Positive coefficients / OR>1 indicate higher odds of the focal group winning. "
                  "Interaction means the effect of relative group size depends on contest location; "
                  "marginal effects computed above show the size effect at LocAdv = -1, 0, +1 (in SD units).")
    }

    return {'object': summary_object, 'description': description}