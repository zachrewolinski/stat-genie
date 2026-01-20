def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderView for participants with dyslexia
    from a fitted statsmodels RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": dict with numeric results (combined coef on log_wpm scale,
                  SE, t-stat, p-value, 95% CI, percent change in WPM, names of terms used)
      - "description": plain-language interpretation of these numbers.

    The function handles two cases:
      1) An interaction term between ReaderView and DyslexiaBin is present:
         it computes the simple effect for DyslexiaBin == 1 as
         coef(ReaderView) + coef(ReaderView:DyslexiaBin).
      2) No interaction term present: it falls back to the main ReaderView
         coefficient (and reports this fact).
    """
    import numpy as np
    from math import isfinite
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Basic checks
    if not hasattr(res, 'params') or not hasattr(res, 'cov_params'):
        raise ValueError("model_output does not look like a statsmodels results object with .params and .cov_params()")

    params = res.params
    cov = res.cov_params()

    # Find parameter names for ReaderView and the interaction with DyslexiaBin
    # Look for exact 'ReaderView' and any parameter name that contains both 'ReaderView' and 'DyslexiaBin'
    param_index = list(params.index)

    reader_param = None
    interaction_param = None

    if 'ReaderView' in param_index:
        reader_param = 'ReaderView'
    else:
        # try case variations or other encodings
        for name in param_index:
            if name.strip() == 'ReaderView':
                reader_param = name
                break

    for name in param_index:
        if ('ReaderView' in name) and ('DyslexiaBin' in name):
            interaction_param = name
            break
    # If interaction not found, also check reversed order
    if interaction_param is None:
        for name in param_index:
            if ('DyslexiaBin' in name) and ('ReaderView' in name):
                interaction_param = name
                break

    # If no ReaderView main effect, abort
    if reader_param is None:
        raise ValueError("Could not find a 'ReaderView' parameter in the model parameters. Found parameters: {}".format(param_index))

    # Build linear combination: weight 1 on ReaderView, plus weight 1 on interaction (if present)
    coeff = params.get(reader_param, np.nan)
    used_terms = [reader_param]

    if interaction_param is not None and interaction_param in params.index:
        coeff_inter = params[interaction_param]
        combined_coef = float(coeff + coeff_inter)
        used_terms.append(interaction_param)
        # Variance of sum = var(a)+var(b)+2cov(a,b)
        try:
            var_reader = float(cov.loc[reader_param, reader_param])
            var_inter = float(cov.loc[interaction_param, interaction_param])
            covar = float(cov.loc[reader_param, interaction_param])
            combined_var = var_reader + var_inter + 2.0 * covar
        except Exception:
            # If covariance lookup fails, fallback to NaN
            combined_var = np.nan
    else:
        # No interaction: use main effect only
        combined_coef = float(coeff)
        try:
            combined_var = float(cov.loc[reader_param, reader_param])
        except Exception:
            combined_var = np.nan

    # Standard error
    se = float(np.sqrt(combined_var)) if (isfinite(combined_var) and combined_var >= 0) else np.nan

    # t-stat and p-value: use df_resid if available for t-distribution; otherwise normal approx
    t_stat = float(combined_coef / se) if (isfinite(se) and se != 0) else np.nan

    # degrees of freedom: try to be conservative and use residual df if present
    df_resid = None
    try:
        df_resid = float(getattr(res, 'df_resid', np.nan))
    except Exception:
        df_resid = None

    if stats is not None:
        if df_resid is not None and isfinite(df_resid) and df_resid > 0:
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df_resid))
            t_crit = float(stats.t.ppf(1.0 - 0.025, df_resid))
        else:
            # normal approximation
            p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
            t_crit = float(stats.norm.ppf(0.975))
    else:
        # scipy not available: normal approx via numpy
        from math import erf, sqrt
        # normal p-value
        if isfinite(t_stat):
            p_value = 2.0 * 0.5 * (1.0 - (1.0 + erf(abs(t_stat) / sqrt(2.0))) / 2.0)  # same as 2*norm.sf
            t_crit = 1.959963984540054  # approx for 95%
        else:
            p_value = np.nan
            t_crit = 1.959963984540054

    # 95% CI on log scale
    if isfinite(se) and isfinite(t_crit):
        ci_low = float(combined_coef - t_crit * se)
        ci_high = float(combined_coef + t_crit * se)
    else:
        ci_low = ci_high = np.nan

    # Percent change in WPM (approx): exp(coef) - 1
    try:
        pct_change = float((np.exp(combined_coef) - 1.0) * 100.0)
    except Exception:
        pct_change = np.nan

    result_obj = {
        'combined_coef_log_wpm': combined_coef,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        '95%_CI_log_wpm': (ci_low, ci_high),
        'percent_change_in_wpm': pct_change,
        'terms_used': used_terms,
        'notes': (
            "Combined effect computed as ReaderView main effect"
            + ((" + " + interaction_param) if (interaction_param is not None) else "")
            + " (i.e., effect of activating ReaderView when DyslexiaBin==1)."
        )
    }

    # Construct a concise description
    if interaction_param is not None:
        desc = (
            "Estimated effect of enabling Reader View for participants with dyslexia (DyslexiaBin=1):\n"
            f"- Linear combination used: {reader_param} + {interaction_param}\n"
            f"- Coefficient on log(WPM): {combined_coef:.4f} (SE = {se:.4f}), t = {t_stat:.2f}, p = {p_value:.3g}\n"
            f"- 95% CI on log(WPM): [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"- Corresponding approximate percent change in WPM: {pct_change:.2f}%\n\n"
            "Interpretation: the coefficient is the estimated difference in log(words-per-minute) when ReaderView is activated for dyslexic readers. "
            "Exponentiating (exp(coef)-1) gives the multiplicative percent change in raw WPM. "
            "Small p-value (e.g., < 0.05) would indicate evidence that ReaderView affects reading speed for dyslexic readers. "
            "These estimates use the model's reported covariance matrix (e.g., clustered standard errors if the model was fit that way)."
        )
    else:
        desc = (
            "No interaction term between ReaderView and DyslexiaBin was found in the model. "
            "Falling back to the main ReaderView effect (applies equally across DyslexiaBin).\n"
            f"- Coefficient on log(WPM) for ReaderView: {combined_coef:.4f} (SE = {se:.4f}), t = {t_stat:.2f}, p = {p_value:.3g}\n"
            f"- 95% CI on log(WPM): [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"- Corresponding approximate percent change in WPM: {pct_change:.2f}%\n\n"
            "Interpretation: because the model does not include an interaction, this estimated effect is the average effect of ReaderView across dyslexic and non-dyslexic readers. "
            "If you intended to test whether the effect differs for dyslexic readers specifically, please include the interaction term ReaderView*DyslexiaBin in the model."
        )

    return {"object": result_obj, "description": desc}