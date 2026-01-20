def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and confidence intervals for the
    predictors of interest (masfem_z and gender_female) from the provided model_output dict,
    and provides a brief interpretation about whether results support the hypothesis that
    more feminine hurricane names are associated with different fatalities.

    Returns:
      {
        "object": {
          "ols": {
            "masfem_z": {coef, se, p_value, ci_lower, ci_upper, approx_pct_change_on_(alldeaths+1)_per_unit},
            "gender_female": { ... }
          },
          "nb": {
            "masfem_z": {coef, se, p_value, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, approx_pct_change_in_rate_per_unit},
            "gender_female": { ... },
          }
        },
        "description": "Brief summary and verdict about the hypothesis..."
      }
    """
    import numpy as np

    required_models = ['ols', 'nb']
    for m in required_models:
        if m not in model_output:
            raise ValueError(f"model_output must contain '{m}' key")

    predict_vars = ['masfem_z', 'gender_female']
    results = {'ols': {}, 'nb': {}}

    # Helper to safely extract conf_int row
    def _get_conf_int(res, name):
        try:
            # Many statsmodels result objects return a DataFrame with index labels equal to param names
            ci = res.conf_int().loc[name]
            return float(ci[0]), float(ci[1])
        except Exception:
            # fallback if conf_int returns array-like without index
            try:
                ci_arr = res.conf_int()
                # find position of param in params index
                idx = list(res.params.index).index(name)
                return float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                return (np.nan, np.nan)

    # Extract from OLS
    ols = model_output['ols']
    for var in predict_vars:
        if var not in ols.params.index:
            # variable might be absent due to perfect multicollinearity or removal
            results['ols'][var] = {
                'present': False,
                'note': f"{var} not found in OLS model parameters"
            }
            continue
        coef = float(ols.params[var])
        se = float(ols.bse[var])
        p = float(ols.pvalues[var]) if hasattr(ols, 'pvalues') else np.nan
        ci_lower, ci_upper = _get_conf_int(ols, var)
        # Interpretation for log outcome: approximate percent change in (alldeaths+1)
        # per 1 unit increase in predictor: (exp(coef)-1)*100
        try:
            pct_change = (np.exp(coef) - 1.0) * 100.0
        except Exception:
            pct_change = np.nan

        results['ols'][var] = {
            'present': True,
            'coef': coef,
            'se': se,
            'p_value': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'approx_pct_change_on_(alldeaths+1)_per_unit': pct_change
        }

    # Extract from NB (GLM)
    nb = model_output['nb']
    for var in predict_vars:
        if var not in nb.params.index:
            results['nb'][var] = {
                'present': False,
                'note': f"{var} not found in NB/GLM model parameters"
            }
            continue
        coef = float(nb.params[var])
        se = float(nb.bse[var])
        # statsmodels GLM results sometimes use z-statistics and pvalues attribute
        p = float(nb.pvalues[var]) if hasattr(nb, 'pvalues') else np.nan
        ci_lower, ci_upper = _get_conf_int(nb, var)
        # Incident Rate Ratio (IRR) and CI
        try:
            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower))
            irr_ci_upper = float(np.exp(ci_upper))
            pct_change_irr = (irr - 1.0) * 100.0
        except Exception:
            irr = irr_ci_lower = irr_ci_upper = pct_change_irr = np.nan

        # Ensure consistent key naming with OLS entries: include 'coef' as the log-scale coefficient
        results['nb'][var] = {
            'present': True,
            'coef': coef,  # log(IRR)
            'se': se,
            'p_value': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'irr': irr,
            'irr_ci_lower': irr_ci_lower,
            'irr_ci_upper': irr_ci_upper,
            'approx_pct_change_in_rate_per_unit': pct_change_irr
        }

    # Make a short verdict about the hypothesis (direction: more feminine -> more fatalities)
    # We'll look at masfem_z in both models: positive coef + p < .05 counts as supporting evidence.
    verdict_notes = []
    pos_sig_count = 0
    neg_sig_count = 0
    models_considered = 0
    for m in ['ols', 'nb']:
        entry = results[m].get('masfem_z')
        if not entry or not entry.get('present', False):
            verdict_notes.append(f"masfem_z not available in {m} model.")
            continue
        models_considered += 1
        coef = entry.get('coef', np.nan)
        p = entry.get('p_value', np.nan)
        if np.isfinite(coef) and np.isfinite(p):
            if (coef > 0) and (p < 0.05):
                pos_sig_count += 1
                verdict_notes.append(f"{m}: masfem_z coef = {coef:.4f}, p = {p:.3g} (positive & significant).")
            elif (coef < 0) and (p < 0.05):
                neg_sig_count += 1
                verdict_notes.append(f"{m}: masfem_z coef = {coef:.4f}, p = {p:.3g} (negative & significant).")
            else:
                verdict_notes.append(f"{m}: masfem_z coef = {coef:.4f}, p = {p:.3g} (not statistically significant).")
        else:
            verdict_notes.append(f"{m}: masfem_z stats not finite or not available.")

    # Derive final conclusion
    if models_considered == 0:
        conclusion = "masfem_z was not available in either model; cannot evaluate the hypothesis."
    else:
        if pos_sig_count == models_considered and pos_sig_count > 0:
            conclusion = ("Consistent evidence across both models: more feminine names (higher masfem_z) "
                          "are associated with greater fatalities. This supports the hypothesis.")
        elif pos_sig_count >= 1:
            conclusion = ("Partial/weak evidence: at least one model shows a statistically significant positive "
                          "association between name femininity and fatalities, which is in the hypothesized direction, "
                          "but results are not consistent across models.")
        elif neg_sig_count >= 1:
            conclusion = ("Evidence against the hypothesis: at least one model shows a statistically significant negative "
                          "association (more feminine names associated with fewer fatalities).")
        else:
            conclusion = ("No statistically significant evidence that masfem_z is associated with fatalities in either model. "
                          "The hypothesis is not supported by these results.")

    # Brief summary describing the key extracted numbers and meaning
    description_lines = [
        "Extracted statistics for predictors (masfem_z and gender_female) from both models (OLS on log_deaths and NB GLM on counts).",
        "For OLS on log_deaths, coefficients reflect change in log(alldeaths + 1); approximate percent change = (exp(coef)-1)*100.",
        "For NB GLM, coefficients are on the log scale; IRR = exp(coef) gives multiplicative change in expected death counts per unit increase.",
        "",
        "Per-variable results:",
        *verdict_notes,
        "",
        "Final conclusion:",
        conclusion
    ]
    description = "\n".join(description_lines)

    return {"object": results, "description": description}