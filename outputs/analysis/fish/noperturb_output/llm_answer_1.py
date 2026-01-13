def extract_final_answer(model_output):
    """
    Extracts key statistics from the model_output dict produced by the modeling function.
    Returns a dictionary with keys:
      - "object": dict with numeric results (coefficients, p-values, IRRs, CIs, significance flags, model used, dispersion, n_obs, mean_rate_per_hour)
      - "description": human-readable summary interpreting the main independent variables (livebait and camper)
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the modeling function.")
    # Choose model: prefer negative binomial if fitted (handles overdispersion), otherwise use poisson
    nb = model_output.get('negative_binomial', None)
    poisson = model_output.get('poisson', None)
    dispersion = float(model_output.get('dispersion', np.nan))
    n_obs = int(model_output.get('n_obs', -1)) if 'n_obs' in model_output else None
    mean_rate_per_hour = float(model_output.get('mean_rate_per_hour', np.nan)) if 'mean_rate_per_hour' in model_output else None

    if nb is not None:
        res = nb
        model_name = 'negative_binomial'
    elif poisson is not None:
        res = poisson
        model_name = 'poisson'
    else:
        raise ValueError("No fitted model found in model_output (expected keys 'negative_binomial' or 'poisson').")

    # Extract coefficient table
    try:
        params = res.params         # pandas Series
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()      # DataFrame: columns [0,1] or ['lower','upper']
    except Exception as e:
        raise RuntimeError(f"Failed to extract results from model object: {e}")

    # Variables of interest
    vars_of_interest = ['livebait', 'camper']
    results = {}
    for var in vars_of_interest:
        if var in params.index:
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else None
            pval = float(pvalues[var]) if var in pvalues.index else None
            # confidence interval
            if var in conf.index:
                ci_lower = float(conf.loc[var, 0])
                ci_upper = float(conf.loc[var, 1])
            else:
                # fallback if conf is differently labeled
                try:
                    ci_row = conf.loc[var]
                    ci_lower = float(ci_row.iloc[0])
                    ci_upper = float(ci_row.iloc[1])
                except Exception:
                    ci_lower = None
                    ci_upper = None
            # Incidence rate ratio (IRR) and CI
            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
            significant = (pval is not None) and (pval < 0.05)
            results[var] = {
                'coefficient': coef,
                'std_error': se,
                'p_value': pval,
                'conf_int_coef_lower': ci_lower,
                'conf_int_coef_upper': ci_upper,
                'incidence_rate_ratio': irr,
                'irr_ci_lower': irr_ci_lower,
                'irr_ci_upper': irr_ci_upper,
                'significant_at_0.05': bool(significant)
            }
        else:
            results[var] = {
                'error': f"Variable '{var}' not found in model parameters."
            }

    # Add model-level info
    final_object = {
        'model_used': model_name,
        'dispersion': float(dispersion) if not np.isnan(dispersion) else None,
        'n_obs': n_obs,
        'mean_rate_per_hour': mean_rate_per_hour,
        'variables': results
    }

    # Prepare human-readable description
    def interpret(varname, stats):
        if 'error' in stats:
            return f"{varname}: {stats['error']}"
        irr = stats['incidence_rate_ratio']
        pval = stats['p_value']
        ci_low = stats['irr_ci_lower']
        ci_high = stats['irr_ci_upper']
        sig = stats['significant_at_0.05']
        direction = "increase" if irr > 1 else ("decrease" if irr < 1 else "no change")
        pct_change = (irr - 1) * 100.0
        pct_str = f"{pct_change:.1f}% {direction}"
        sig_str = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"
        ci_str = f"95% CI for IRR: [{ci_low:.3f}, {ci_high:.3f}]" if (ci_low is not None and ci_high is not None) else "CI unavailable"
        return (f"{varname}: IRR = {irr:.3f} ({ci_str}), p = {pval:.3g}. "
                f"This implies a {pct_str} in the catch rate per hour when {varname} = 1, "
                f"and this effect is {sig_str}.")

    desc_lines = []
    desc_lines.append(f"Model used for inference: {final_object['model_used']}. Overdispersion = {final_object['dispersion']:.3f} (threshold >1.5 suggests overdispersion).")
    desc_lines.append(f"Sample size: n = {final_object['n_obs']}. Mean observed catch rate per hour = {final_object['mean_rate_per_hour']:.3f}.")
    # Interpret each variable
    for v in vars_of_interest:
        desc_lines.append(interpret(v, results[v]))
    description = " ".join(desc_lines)

    return {
        "object": final_object,
        "description": description
    }