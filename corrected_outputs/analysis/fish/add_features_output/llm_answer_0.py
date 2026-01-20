def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted model output dict returned by the modeling function.

    Returns a dictionary with keys:
      - "object": a dict with:
          - model_used: which model object was used for inference ('negbin_model' preferred when present)
          - overdispersion: the Pearson dispersion measure from the original fit
          - aic: AIC(s) for model(s) if available
          - variables: per-variable stats for ['livebait','camper','total_people'] if present
              each variable entry contains:
                - coef: estimated coefficient (log rate change)
                - pvalue: p-value
                - ci_lower, ci_upper: 95% CI on coefficient
                - irr: exp(coef) (incidence rate ratio = multiplicative change in fish/hour)
                - irr_ci_lower, irr_ci_upper: 95% CI for IRR
                - significant: True if pvalue < 0.05
      - "description": short plain-language interpretation of the extracted statistics
    """
    import numpy as np

    # Decide which model to use for inference: prefer Negative Binomial if available (handles overdispersion)
    model_key = 'negbin_model' if 'negbin_model' in model_output else 'poisson_model'
    if model_key not in model_output:
        raise ValueError("model_output does not contain 'poisson_model' or 'negbin_model'.")

    results = model_output[model_key]
    # Extract numeric summaries
    params = results.params          # pandas Series-like
    pvalues = results.pvalues
    # conf_int may be a DataFrame or ndarray-like
    try:
        conf = results.conf_int()    # typically DataFrame with param names as index
    except Exception:
        # fallback: compute approximate conf int using normal approx if bse available
        bse = results.bse
        z = 1.96
        conf = None
        # we'll handle missing conf below

    def get_conf_interval(var):
        # Return (lower, upper) as floats for parameter var
        # Try conf.loc[var], else try conf.iloc based on matching name, else use normal approx
        if 'conf' in locals() and conf is not None:
            try:
                row = conf.loc[var]
                low, high = float(row[0]), float(row[1])
                return low, high
            except Exception:
                # try searching index for a match
                try:
                    idx = list(conf.index).index(var)
                    row = conf.iloc[idx]
                    return float(row[0]), float(row[1])
                except Exception:
                    pass
        # fallback: normal approx
        try:
            se = float(results.bse[var])
            est = float(params[var])
            low = est - 1.96 * se
            high = est + 1.96 * se
            return low, high
        except Exception:
            return (np.nan, np.nan)

    # Variables of interest
    variables = ['livebait', 'camper', 'total_people']
    var_stats = {}
    for var in variables:
        if var in params.index:
            coef = float(params[var])
            pval = float(pvalues[var]) if var in pvalues.index else np.nan
            ci_low, ci_high = get_conf_interval(var)
            irr = float(np.exp(coef))
            irr_ci_low = float(np.exp(ci_low)) if not (ci_low is None or np.isnan(ci_low)) else np.nan
            irr_ci_high = float(np.exp(ci_high)) if not (ci_high is None or np.isnan(ci_high)) else np.nan
            var_stats[var] = {
                'coef': coef,
                'pvalue': pval,
                'ci_lower': float(ci_low) if not np.isnan(ci_low) else None,
                'ci_upper': float(ci_high) if not np.isnan(ci_high) else None,
                'irr': irr,
                'irr_ci_lower': irr_ci_low,
                'irr_ci_upper': irr_ci_high,
                'significant': bool((not np.isnan(pval)) and (pval < 0.05))
            }

    # Collect AIC(s) and overdispersion
    aics = {}
    if 'poisson_aic' in model_output:
        aics['poisson_aic'] = float(model_output['poisson_aic'])
    if 'negbin_aic' in model_output:
        aics['negbin_aic'] = float(model_output['negbin_aic'])
    overdispersion = float(model_output.get('overdispersion', np.nan))

    object_out = {
        'model_used': model_key,
        'overdispersion': overdispersion,
        'aic': aics,
        'variables': var_stats
    }

    # Compose a concise description
    # Note: interpret coefficients from the chosen model: coef = log(rate ratio; irr = multiplicative change in fish/hour)
    desc_lines = []
    desc_lines.append(f"Using '{model_key}' for inference (negative binomial preferred when present).")
    desc_lines.append(f"Overdispersion measure: {overdispersion:.3g}.")
    if aics:
        aic_parts = [f"{k}={v:.1f}" for k, v in aics.items()]
        desc_lines.append("AIC(s): " + ", ".join(aic_parts) + ".")
    for var, s in var_stats.items():
        sig_text = "statistically significant" if s['significant'] else "not statistically significant"
        desc_lines.append(
            f"{var}: coef={s['coef']:.3f} (log rate), IRR={s['irr']:.3f} "
            f"95%CI_IRR=[{s['irr_ci_lower']:.3f}, {s['irr_ci_upper']:.3f}], p={s['pvalue']:.3g} -> {sig_text}."
        )
    description = " ".join(desc_lines)

    return {
        "object": object_out,
        "description": description
    }