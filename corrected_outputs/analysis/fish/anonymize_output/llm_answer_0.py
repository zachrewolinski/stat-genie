def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (or a dict
    containing it under key 'results') and produce interpretable rate-per-hour
    results for the fishing model.

    Returns a dict with keys:
      - "object": dict mapping each model term to extracted stats (coef, se, p, CI,
                  rate ratio, RR CI, significant boolean)
      - "description": short plain-language interpretation of the findings,
                       including baseline rate per hour and which predictors
                       significantly change the catch rate.
    """
    import numpy as np
    import pandas as pd

    # Accept either the raw results object or the dict produced by the modeling code
    if isinstance(model_output, dict):
        results = model_output.get('results')
        model_type = model_output.get('model_type', None)
        overdispersion = model_output.get('overdispersion', None)
    else:
        results = model_output
        model_type = None
        overdispersion = None

    if results is None:
        raise ValueError("No fitted model results found in model_output.")

    # Extract coefficient table
    params = results.params.copy()
    bse = results.bse.copy()
    pvalues = results.pvalues.copy()
    try:
        conf = results.conf_int()
        conf.columns = ['ci_lower', 'ci_upper']
    except Exception:
        # If conf_int fails for some reason, fill with NaNs
        conf = pd.DataFrame(index=params.index, data={'ci_lower': np.nan, 'ci_upper': np.nan})

    # Exponentiated coefficients are rate ratios because the model uses log link
    rr = np.exp(params)
    rr_ci_lower = np.exp(conf['ci_lower'])
    rr_ci_upper = np.exp(conf['ci_upper'])

    # Build summary table
    rows = {}
    for idx in params.index:
        coef = float(params.loc[idx])
        se = float(bse.loc[idx]) if idx in bse.index else np.nan
        p = float(pvalues.loc[idx]) if idx in pvalues.index else np.nan
        ci_l = float(conf.loc[idx, 'ci_lower']) if idx in conf.index else np.nan
        ci_u = float(conf.loc[idx, 'ci_upper']) if idx in conf.index else np.nan
        rate_ratio = float(rr.loc[idx])
        rr_l = float(rr_ci_lower.loc[idx]) if idx in rr_ci_lower.index else np.nan
        rr_u = float(rr_ci_upper.loc[idx]) if idx in rr_ci_upper.index else np.nan
        significant = (not np.isnan(p)) and (p < 0.05)
        rows[idx] = {
            'coef_log_rate': coef,
            'se': se,
            'p_value': p,
            'ci_log_rate_2.5%': ci_l,
            'ci_log_rate_97.5%': ci_u,
            'rate_ratio (per-hour multiplier)': rate_ratio,
            'rr_2.5%': rr_l,
            'rr_97.5%': rr_u,
            'significant (p<0.05)': bool(significant),
            'percent_change_in_rate (%)': (rate_ratio - 1.0) * 100.0
        }

    # Baseline rate per hour: exp(const) if const present; this is rate when covariates=0
    baseline_info = {}
    if 'const' in params.index:
        baseline_rate = float(np.exp(params['const']))
        baseline_ci = (float(np.exp(conf.loc['const', 'ci_lower'])),
                       float(np.exp(conf.loc['const', 'ci_upper'])))
        baseline_info = {
            'baseline_rate_per_hour': baseline_rate,
            'baseline_rate_ci_lower': baseline_ci[0],
            'baseline_rate_ci_upper': baseline_ci[1],
            'note': ("Baseline rate corresponds to covariates = 0 (LiveBait=0, Camper=0, "
                     "Adults=0, Children=0). If that combination is not realistic, compute "
                     "predicted rate for realistic covariate values by summing the "
                     "appropriate coefficients and exponentiating.")
        }
    else:
        baseline_info = {'baseline_rate_per_hour': None, 'note': 'No intercept (const) found in model params.'}

    # Short textual interpretation of the most important results
    # Identify statistically significant predictors (p < 0.05)
    sig_terms = [t for t, v in rows.items() if v['significant (p<0.05)'] and t != 'const']
    def term_summary(name):
        v = rows[name]
        pct = v['percent_change_in_rate (%)']
        return f"{name}: RR={v['rate_ratio (per-hour multiplier)']:.3f} (CI {v['rr_2.5%']:.3f}–{v['rr_97.5%']:.3f}), p={v['p_value']:.3g}, ~{pct:.1f}% change"

    if sig_terms:
        sig_texts = "; ".join(term_summary(t) for t in sig_terms)
        significance_text = f"Significant predictors: {sig_texts}."
    else:
        significance_text = "No predictor besides the intercept was statistically significant at p<0.05."

    description = (
        f"Model type: {model_type}. Overdispersion metric (deviance/df): {overdispersion}.\n"
        f"Baseline expected catch rate per hour (covariates=0): "
        f"{baseline_info.get('baseline_rate_per_hour'):.3f} fish/hour "
        f"(CI {baseline_info.get('baseline_rate_ci_lower'):.3f}–{baseline_info.get('baseline_rate_ci_upper'):.3f})\n"
        f"{baseline_info.get('note')}\n"
        f"{significance_text}\n"
        "Interpretation notes: coefficients are on the log-rate scale. Exponentiated coefficients "
        "('rate_ratio') multiply the expected fish-per-hour rate. For interaction terms (LiveBait_Camper), "
        "interpretation is multiplicative on top of the individual LiveBait and Camper effects."
    )

    return {
        "object": {
            "model_type": model_type,
            "overdispersion": overdispersion,
            "term_table": rows,
            "baseline": baseline_info
        },
        "description": description
    }