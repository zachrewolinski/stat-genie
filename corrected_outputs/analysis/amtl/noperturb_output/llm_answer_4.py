def extract_final_answer(model_output):
    """
    Extracts statistics for the 'is_human' effect from the model output produced by the
    provided modeling function and returns a concise object and a human-readable description.

    Returns:
      {
        "object": {
          "is_human_coef": float (log-odds coefficient),
          "p_value": float,
          "odds_ratio": float,
          "odds_ratio_ci": [float_lower, float_upper],
          "significant": bool  # True if p_value < 0.05
        },
        "description": str  # brief interpretation in the context of the task
      }
    """
    import numpy as np
    import math

    # Defensive extraction helpers
    def safe_get(d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else default

    # Look for robust_result first, then coef_table, then glm_result
    robust = safe_get(model_output, 'robust_result', None)
    coef_table = safe_get(model_output, 'coef_table', None)
    glm_res = safe_get(model_output, 'glm_result', None)
    # Also check pre-computed odds ratio fields if present
    or_est = safe_get(model_output, 'is_human_odds_ratio', None)
    or_ci = safe_get(model_output, 'is_human_odds_ratio_ci', None)

    param = 'is_human'
    coef = None
    pval = None
    ci_lower = None
    ci_upper = None

    # Try robust_result (preferred)
    try:
        if robust is not None and hasattr(robust, 'params') and param in robust.params.index:
            coef = float(robust.params[param])
            # p-value available on robust wrapper
            if hasattr(robust, 'pvalues') and param in robust.pvalues.index:
                pval = float(robust.pvalues[param])
            # confidence interval on coefficient (log-odds scale)
            try:
                ci_df = robust.conf_int()
                if param in ci_df.index:
                    ci_lower = float(ci_df.loc[param][0])
                    ci_upper = float(ci_df.loc[param][1])
            except Exception:
                # fall back to coef +/- 1.96*std_err if available
                if hasattr(robust, 'bse') and param in robust.bse.index:
                    se = float(robust.bse[param])
                    z = 1.96
                    ci_lower = coef - z * se
                    ci_upper = coef + z * se
    except Exception:
        pass

    # If robust not available, try coef_table (DataFrame-like)
    if coef is None and coef_table is not None:
        try:
            # coef_table may be a DataFrame; try to read columns
            if param in coef_table.index:
                row = coef_table.loc[param]
                coef = float(row.get('coef', row.get('Coef.', None) ))
                pval = float(row.get('P>|z|', row.get('P>|t|', None)))
                # compute CI if std_err exists
                std_err = row.get('std_err', row.get('Std.Err', None))
                if std_err is not None:
                    se = float(std_err)
                    z = 1.96
                    ci_lower = coef - z * se
                    ci_upper = coef + z * se
        except Exception:
            pass

    # As a last resort, use glm_result params and cov_params if available
    if coef is None and glm_res is not None:
        try:
            params = glm_res.params
            if param in params.index:
                coef = float(params[param])
                # p-value from glm_res if available
                try:
                    pval = float(glm_res.pvalues[param])
                except Exception:
                    pval = None
                # try to compute std err from covariance matrix
                try:
                    cov = glm_res.cov_params()
                    se = float(np.sqrt(np.maximum(np.diag(cov), 0.0))[list(params.index).index(param)])
                    z = 1.96
                    ci_lower = coef - z * se
                    ci_upper = coef + z * se
                except Exception:
                    pass
        except Exception:
            pass

    # If odds ratio and CI were pre-computed in model_output, prefer those for OR/CI
    if or_est is not None:
        try:
            odds_ratio = float(or_est)
        except Exception:
            odds_ratio = None
    else:
        odds_ratio = None
    if or_ci is not None and isinstance(or_ci, (list, tuple)) and len(or_ci) == 2:
        try:
            or_ci_lower = float(or_ci[0])
            or_ci_upper = float(or_ci[1])
        except Exception:
            or_ci_lower = or_ci_upper = None
    else:
        or_ci_lower = or_ci_upper = None

    # If we have coef but not odds ratio, compute it
    if odds_ratio is None and coef is not None:
        try:
            odds_ratio = float(math.exp(coef))
        except Exception:
            odds_ratio = None

    # If we have CI on coef but not on OR, exponentiate
    if (or_ci_lower is None or or_ci_upper is None) and (ci_lower is not None and ci_upper is not None):
        try:
            or_ci_lower = float(math.exp(ci_lower))
            or_ci_upper = float(math.exp(ci_upper))
        except Exception:
            or_ci_lower = or_ci_upper = None

    # Final significance decision
    significant = None
    if pval is not None:
        significant = bool(pval < 0.05)

    # Round numeric outputs to sensible precision for readability
    def _safe_round(x, digits=6):
        try:
            return float(np.round(x, digits))
        except Exception:
            return x

    result_object = {
        'is_human_coef': _safe_round(coef, 6) if coef is not None else None,
        'p_value': _safe_round(pval, 6) if pval is not None else None,
        'odds_ratio': _safe_round(odds_ratio, 6) if odds_ratio is not None else None,
        'odds_ratio_ci': [
            _safe_round(or_ci_lower, 6) if or_ci_lower is not None else None,
            _safe_round(or_ci_upper, 6) if or_ci_upper is not None else None
        ],
        'significant': significant
    }

    # Human-readable interpretation
    if coef is None:
        description = ("Could not locate 'is_human' coefficient in the provided model output. "
                       "Please supply model output containing 'robust_result', 'coef_table', or 'glm_result'.")
    else:
        # Compose a concise interpretation
        or_str = f"OR = {result_object['odds_ratio']}"
        ci_vals = result_object['odds_ratio_ci']
        if ci_vals and ci_vals[0] is not None and ci_vals[1] is not None:
            or_str += f" (95% CI: {ci_vals[0]}–{ci_vals[1]})"
        pstr = f"p = {result_object['p_value']}" if result_object['p_value'] is not None else "p-value unavailable"
        sig_str = "statistically significant" if significant else ("not statistically significant" if significant is not None else "significance unknown")
        description = (f"The estimated effect of being a modern human on AMTL is a log-odds coefficient = {result_object['is_human_coef']}. "
                       f"This corresponds to {or_str}; {pstr}. "
                       f"Conclusion: modern humans have { 'higher' if coef > 0 else 'lower' } odds of AMTL compared to the non-human primate genera after controlling for age (age_z), sex probability (prob_male), and tooth class, "
                       f"with standard errors clustered by specimen. The effect is {sig_str} at α = 0.05.")

    return {"object": result_object, "description": description}