def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels logistic regression result
    about the effect of relative group size (LogSizeRatio) and its interaction
    with contest location (AtFocalHome) on the probability the focal group wins.

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results (coefficients, SEs, p-values,
                    odds ratios, 95% CIs) for:
                    * LogSizeRatio (main effect, i.e., effect when AtFocalHome=0)
                    * AtFocalHome (main effect)
                    * Interaction (LogSizeRatio:AtFocalHome)
                    * Combined LogSizeRatio effect when AtFocalHome=1 (main+interaction)
                    Also returns the covariance matrix used for combined SE calculation.
        - "description": short plain-language interpretation of the results.
    """
    import numpy as np
    import math

    res = model_output

    # Extract parameter names and arrays/series
    try:
        params = res.params  # pandas Series
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int(alpha=0.05)  # DataFrame with two columns
        cov = res.cov_params()
    except Exception as e:
        raise ValueError(f"Unable to extract required attributes from model_output: {e}")

    # Helper to find parameter name robustly
    def find_param_token(*tokens, exclude_colon=False):
        """Find a parameter name that contains all tokens (in any order).
           If exclude_colon=True, skip names that contain ':' (interaction names)."""
        for name in params.index:
            if exclude_colon and ':' in name:
                continue
            if all(tok in name for tok in tokens):
                return name
        return None

    # Find main and interaction parameter names
    main_log_name = find_param_token('LogSizeRatio', exclude_colon=True) or find_param_token('LogSizeRatio')
    af_name = find_param_token('AtFocalHome', exclude_colon=True) or find_param_token('AtFocalHome')
    inter_name = None
    # interaction should contain both tokens and a ':' typically
    for name in params.index:
        if 'LogSizeRatio' in name and 'AtFocalHome' in name and name != main_log_name:
            inter_name = name
            break

    if main_log_name is None:
        raise KeyError("Could not locate a parameter corresponding to 'LogSizeRatio' in model_output.params")

    # Extract values for main effect
    beta_log = float(params[main_log_name])
    se_log = float(bse[main_log_name]) if main_log_name in bse.index else float(np.nan)
    p_log = float(pvalues[main_log_name]) if main_log_name in pvalues.index else float(np.nan)
    ci_log = tuple(conf.loc[main_log_name]) if main_log_name in conf.index else (np.nan, np.nan)

    # Extract AtFocalHome main effect if present
    if af_name is not None:
        beta_af = float(params[af_name])
        se_af = float(bse[af_name]) if af_name in bse.index else float(np.nan)
        p_af = float(pvalues[af_name]) if af_name in pvalues.index else float(np.nan)
        ci_af = tuple(conf.loc[af_name]) if af_name in conf.index else (np.nan, np.nan)
    else:
        beta_af = se_af = p_af = np.nan
        ci_af = (np.nan, np.nan)

    # Extract interaction effect if present
    if inter_name is not None:
        beta_inter = float(params[inter_name])
        se_inter = float(bse[inter_name]) if inter_name in bse.index else float(np.nan)
        p_inter = float(pvalues[inter_name]) if inter_name in pvalues.index else float(np.nan)
        ci_inter = tuple(conf.loc[inter_name]) if inter_name in conf.index else (np.nan, np.nan)
    else:
        # No explicit interaction term found (shouldn't happen given the formula)
        beta_inter = se_inter = p_inter = np.nan
        ci_inter = (np.nan, np.nan)

    # Combined effect of LogSizeRatio when AtFocalHome = 1: beta_log + beta_inter
    beta_combined = beta_log + (beta_inter if not math.isnan(beta_inter) else 0.0)

    # Compute SE for combined effect using covariance matrix if available
    se_combined = float(np.nan)
    p_combined = float(np.nan)
    ci_combined = (float(np.nan), float(np.nan))
    try:
        # cov should be a DataFrame with indices matching params.index
        if inter_name is not None and main_log_name in cov.index and inter_name in cov.index:
            var_main = float(cov.loc[main_log_name, main_log_name])
            var_inter = float(cov.loc[inter_name, inter_name])
            covar = float(cov.loc[main_log_name, inter_name])
            var_sum = var_main + var_inter + 2.0 * covar
            se_combined = math.sqrt(max(var_sum, 0.0))
        else:
            # fallback to naive sum of variances (less correct)
            se_combined = math.sqrt(max(se_log**2 + (se_inter**2 if not math.isnan(se_inter) else 0.0), 0.0))

        # z and two-sided p-value using normal approximation
        if se_combined > 0:
            z = beta_combined / se_combined
            # p-value using math.erfc for robustness (avoids scipy dependency)
            p_combined = float(math.erfc(abs(z) / math.sqrt(2.0)))
            # 95% CI on log-odds scale and convert to odds ratio scale
            ci_low = beta_combined - 1.96 * se_combined
            ci_high = beta_combined + 1.96 * se_combined
            ci_combined = (ci_low, ci_high)
        else:
            p_combined = float(np.nan)
            ci_combined = (float(np.nan), float(np.nan))
    except Exception:
        # keep NaNs if something goes wrong with cov extraction
        se_combined = float(np.nan)
        p_combined = float(np.nan)
        ci_combined = (float(np.nan), float(np.nan))

    # Compute odds ratios and 95% CIs (exp of coefficients)
    def or_and_ci(beta, se):
        if beta is None or (se is None) or (math.isnan(beta)):
            return (float(np.nan), (float(np.nan), float(np.nan)))
        if se is None or math.isnan(se):
            or_val = math.exp(beta)
            return (or_val, (float(np.nan), float(np.nan)))
        low = beta - 1.96 * se
        high = beta + 1.96 * se
        return (math.exp(beta), (math.exp(low), math.exp(high)))

    or_log, orci_log = or_and_ci(beta_log, se_log)
    or_inter, orci_inter = or_and_ci(beta_inter, se_inter)
    or_combined, orci_combined = or_and_ci(beta_combined, se_combined)

    # Prepare the result object with key values
    results_object = {
        'params': {
            'LogSizeRatio_name': main_log_name,
            'LogSizeRatio_coef': beta_log,
            'LogSizeRatio_se': se_log,
            'LogSizeRatio_p': p_log,
            'LogSizeRatio_CI_95': tuple(ci_log),
            'LogSizeRatio_OR': or_log,
            'LogSizeRatio_OR_CI_95': orci_log,
            'AtFocalHome_name': af_name,
            'AtFocalHome_coef': beta_af,
            'AtFocalHome_se': se_af,
            'AtFocalHome_p': p_af,
            'AtFocalHome_CI_95': tuple(ci_af),
            'Interaction_name': inter_name,
            'Interaction_coef': beta_inter,
            'Interaction_se': se_inter,
            'Interaction_p': p_inter,
            'Interaction_CI_95': tuple(ci_inter),
            'Combined_LogSizeRatio_at_AtFocalHome=1_coef': beta_combined,
            'Combined_SE': se_combined,
            'Combined_p': p_combined,
            'Combined_CI_95': tuple(ci_combined),
            'Combined_OR': or_combined,
            'Combined_OR_CI_95': orci_combined,
        },
        'cov_params_used': cov if hasattr(cov, 'to_dict') else None
    }

    # Compose a concise interpretation
    # Interpret direction: positive beta_log -> higher log-size ratio (focal larger) increases log-odds of winning
    def significance_label(p):
        try:
            if math.isnan(p):
                return "p = NA"
            elif p < 0.001:
                return "p < 0.001"
            else:
                return f"p = {p:.3f}"
        except Exception:
            return "p = NA"

    desc_lines = []
    desc_lines.append(
        f"Main effect (LogSizeRatio; effect when AtFocalHome=0): coef = {beta_log:.3f}, SE = {se_log:.3f}, "
        f"{significance_label(p_log)}. Odds ratio = {or_log:.3f} (95% CI: {orci_log[0]:.3f}–{orci_log[1]:.3f})."
    )
    if inter_name is not None:
        desc_lines.append(
            f"Interaction (LogSizeRatio x AtFocalHome): coef = {beta_inter:.3f}, SE = {se_inter:.3f}, "
            f"{significance_label(p_inter)}. This indicates how the effect of relative size changes when contests are at the focal group's home."
        )
        desc_lines.append(
            f"Combined effect of LogSizeRatio when contest is at focal home (AtFocalHome=1): coef = {beta_combined:.3f}, SE = {se_combined:.3f}, "
            f"{significance_label(p_combined)}. Odds ratio = {or_combined:.3f} (95% CI: {orci_combined[0]:.3f}–{orci_combined[1]:.3f})."
        )
    else:
        desc_lines.append("No interaction term was found in the fitted model output; combined effect cannot be computed.")

    # Short plain-language summary
    # Note: user should inspect p-values to determine statistical significance
    summary = " ".join(desc_lines)

    return {
        "object": results_object,
        "description": summary
    }