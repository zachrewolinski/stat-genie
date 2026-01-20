def extract_final_answer(model_output):
    """
    Extracts key statistics from the model output dictionary produced by the modeling function.
    Returns a dictionary with:
      - "object": a structured dict of numeric results (coefficients, IRRs, 95% CIs, p-values, significance flags)
      - "description": a concise, plain-language interpretation of the main findings (baseline rate and which predictors matter)
    """
    import numpy as np

    # 1. Identify chosen fitted model (prefer negative binomial if selected)
    chosen = None
    chosen_name = model_output.get('chosen_model', None)
    # If the model function stored model objects under explicit keys, pick that first
    if chosen_name == 'negative_binomial' and 'negative_binomial_result' in model_output:
        chosen = model_output['negative_binomial_result']
    elif 'poisson_result' in model_output:
        # fallback to Poisson
        chosen = model_output['poisson_result']
    else:
        # As a last resort, try to use whatever key looks like a results object
        for k in ['negative_binomial_result', 'poisson_result']:
            if k in model_output:
                chosen = model_output[k]
                chosen_name = k
                break

    if chosen is None:
        # If no model object is available, try to use precomputed summary keys
        params = model_output.get('params', None)
        irr = model_output.get('IRR', None)
        conf_exp = model_output.get('conf_int_exp', None)
        pvalues = None
    else:
        # Extract from statsmodels results wrapper
        try:
            params = chosen.params
        except Exception:
            params = model_output.get('params', None)
        try:
            pvalues = chosen.pvalues
        except Exception:
            pvalues = None
        try:
            conf = chosen.conf_int()
            # exponentiate confidence intervals to get IRR CIs
            conf_exp = np.exp(conf)
        except Exception:
            conf_exp = model_output.get('conf_int_exp', None)
        # compute IRRs
        try:
            irr = np.exp(params)
        except Exception:
            irr = model_output.get('IRR', None)

    if params is None or irr is None:
        raise ValueError("Model output does not contain parameters or IRRs; cannot extract results.")

    # Normalize to pandas-like access if necessary
    # Build stats for the typical variables used in the modeling function
    predictor_names = ['const', 'livebait', 'camper', 'persons', 'child']
    stats = {}
    for name in predictor_names:
        if name in params.index:
            coef = float(params[name])
            irr_val = float(irr[name])
            # confidence interval for IRR
            ci = None
            if conf_exp is not None:
                try:
                    # conf_exp may be a DataFrame with 2 columns
                    row = conf_exp.loc[name].values
                    ci = (float(row[0]), float(row[1]))
                except Exception:
                    try:
                        # conf_exp might be a dict or other mapping
                        ci = tuple(map(float, conf_exp[name]))
                    except Exception:
                        ci = None
            # p-value (may be None)
            pval = None
            if pvalues is not None:
                try:
                    pval = float(pvalues[name])
                except Exception:
                    pval = None
            significant = None
            if pval is not None:
                significant = (pval < 0.05)
            stats[name] = {
                'coef': coef,
                'IRR': irr_val,
                'IRR_95CI': ci,
                'pvalue': pval,
                'significant_at_0.05': significant
            }

    # Include overdispersion ratio if present
    overdispersion = model_output.get('overdispersion_ratio', None)

    # Prepare a human-readable summary description
    # Baseline rate per hour: exponentiated intercept (const)
    const_stats = stats.get('const')
    if const_stats is not None:
        baseline_rate = const_stats['IRR']
        baseline_ci = const_stats['IRR_95CI']
        baseline_text = f"Estimated baseline catch rate (reference group, predictors=0): {baseline_rate:.2f} fish/hour"
        if baseline_ci is not None:
            baseline_text += f" (95% CI: {baseline_ci[0]:.2f}–{baseline_ci[1]:.2f})"
    else:
        baseline_text = "Baseline (intercept) not available."

    # Interpret each predictor briefly
    parts = [f"Chosen model: {model_output.get('chosen_model', 'unknown')}.",
             baseline_text]
    for var in ['livebait', 'camper', 'persons', 'child']:
        if var in stats:
            s = stats[var]
            irr_str = f"{s['IRR']:.3f}"
            ci_str = "CI unavailable"
            if s['IRR_95CI'] is not None:
                ci_str = f"{s['IRR_95CI'][0]:.3f}–{s['IRR_95CI'][1]:.3f}"
            pstr = f"p={s['pvalue']:.3f}" if s['pvalue'] is not None else "p-value unavailable"
            signif = "statistically significant" if s['significant_at_0.05'] else "not statistically significant"
            # short substantive interpretation
            if var in ['livebait', 'camper']:
                action = "use of" if var == 'livebait' else "having a camper"
                parts.append(f"{action.capitalize()}: IRR={irr_str} (95% CI {ci_str}), {pstr}; {signif}.")
            else:
                # numeric group-size variables
                parts.append(f"Each additional {('adult' if var=='persons' else 'child')}: IRR={irr_str} (95% CI {ci_str}), {pstr}; {signif}.")
    if overdispersion is not None:
        parts.append(f"Poisson overdispersion ratio (Pearson chi2 / df): {overdispersion:.2f} (NB model selected if > 1.5).")

    description = " ".join(parts)

    # Final object to return with structured numeric outputs and short description
    output_object = {
        'chosen_model': model_output.get('chosen_model'),
        'overdispersion_ratio': overdispersion,
        'baseline_rate_per_hour': const_stats,
        'predictors': {k: v for k, v in stats.items() if k != 'const'}
    }

    return {
        "object": output_object,
        "description": description
    }