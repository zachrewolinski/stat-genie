def extract_final_answer(model_output):
    """
    Extracts key statistics for the femininity of hurricane names from a fitted model object.
    
    Returns a dictionary with:
      - "object": dict mapping each target variable ('MasFem_z', 'FemaleName') to its
                  extracted statistics (coef, se, p-value, 95% CI, exp(coef), exp(CI), significance).
      - "description": human-readable explanation of what the numbers mean in context.
    
    Works with:
      - statsmodels GLMResultsWrapper (e.g., NegativeBinomial)
      - statsmodels OLS results (fallback OLS on LogDeaths)
    """
    import numpy as np
    import pandas as pd

    # Variables of interest
    targets = ['MasFem_z', 'FemaleName']
    out = {}
    
    # Basic availability checks
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = getattr(model_output, 'bse', None)
        conf_int_df = model_output.conf_int()
    except Exception as e:
        raise ValueError(f"Cannot extract parameters from model_output: {e}")
    
    # Determine model type (Negative Binomial GLM with log link vs. OLS on log outcome)
    model_family_name = None
    try:
        fam = getattr(model_output.model, 'family', None)
        model_family_name = fam.__class__.__name__ if fam is not None else None
    except Exception:
        model_family_name = None

    # If it's an OLS fitted on LogDeaths, we will note that separately
    is_glm_nb = (model_family_name is not None) and ('NegativeBinomial' in model_family_name)
    is_ols = not is_glm_nb

    for var in targets:
        if var not in params.index:
            out[var] = {
                'present': False,
                'message': f"Variable '{var}' not present in model parameters."
            }
            continue

        coef = float(params[var])
        se = float(bse[var]) if bse is not None and var in bse.index else None
        pval = float(pvalues[var]) if var in pvalues.index else None

        # Confidence interval (two-sided 95%)
        if var in conf_int_df.index:
            ci_low, ci_high = float(conf_int_df.loc[var, 0]), float(conf_int_df.loc[var, 1])
        else:
            ci_low, ci_high = None, None

        # Exponentiated coefficient and CI (interpretable as rate ratio for NB with log link).
        # For OLS on log outcome, exp(coef) gives an approximate multiplicative factor on (1+Deaths).
        exp_coef = float(np.exp(coef))
        exp_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        exp_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        signif = None
        if pval is not None:
            signif = (pval < 0.05)

        out[var] = {
            'present': True,
            'coef': coef,
            'std_error': se,
            'p_value': pval,
            'ci_95': (ci_low, ci_high),
            'exp_coef': exp_coef,
            'exp_ci_95': (exp_ci_low, exp_ci_high),
            'significant_at_0.05': signif,
            'model_family': model_family_name or 'OLS or unknown'
        }

    # Compose a concise description
    if is_glm_nb:
        model_text = ("Model is a Negative Binomial GLM with log link. Coefficients are on the log-count scale; "
                      "exp(coef) is the multiplicative change in expected death counts per one unit increase in predictor.")
        interpret_note = ("For 'MasFem_z': exp_coef gives the multiplicative change in expected deaths "
                          "per 1 SD increase in the masculinity-femininity index (higher = more feminine).\n"
                          "For 'FemaleName' (binary): exp_coef is the incident rate ratio comparing female-named hurricanes "
                          "to male-named hurricanes (value >1 = higher expected deaths for female names; <1 = lower).")
    else:
        model_text = ("Model appears to be OLS (likely run on log(1+Deaths) as a fallback). "
                      "Coefficients represent additive changes in log(1+Deaths).")
        interpret_note = ("exp(coef) is an approximate multiplicative change in (1+Deaths): "
                          "(exp(coef)-1)*100 roughly approximates percent change in expected deaths per unit change in predictor. "
                          "Interpret with caution because OLS on log(1+Deaths) is an approximation.")

    # Final description string
    description_lines = [
        model_text,
        "",
        "Extracted statistics for variables related to name femininity:",
        ""
    ]
    for var, stats in out.items():
        if not stats['present']:
            description_lines.append(f"- {var}: NOT PRESENT in model.")
            continue
        desc = (f"- {var}: coef={stats['coef']:.4f}, SE={stats['std_error']:.4f} " 
                f"p={stats['p_value']:.4g}, 95% CI=[{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}] "
                f"=> exp(coef)={stats['exp_coef']:.4f}, exp(95% CI)=[{stats['exp_ci_95'][0]:.4f}, {stats['exp_ci_95'][1]:.4f}]. "
                f"Significant@0.05: {stats['significant_at_0.05']}.")
        description_lines.append(desc)
    description_lines.append("")
    description_lines.append(interpret_note)

    description = "\n".join(description_lines)

    return {
        "object": out,
        "description": description
    }