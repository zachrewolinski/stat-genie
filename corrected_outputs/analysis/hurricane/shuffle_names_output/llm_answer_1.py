def extract_final_answer(model_output):
    """
    Extract key statistics for the FemScore_z coefficient from a fitted statsmodels
    RegressionResultsWrapper (expected to be fitted with cov_type='HC3').

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, se, t, p, 95% CI, exponentiated effect)
      - "description": a short plain-language interpretation in the context of the task
    """
    import numpy as np

    res = model_output
    var = 'FemScore_z'

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object.")
    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in the model output parameters.")

    # Extract statistics
    coef = float(res.params[var])
    se = float(res.bse[var]) if hasattr(res, 'bse') and var in res.bse.index else None
    tval = float(res.tvalues[var]) if hasattr(res, 'tvalues') and var in res.tvalues.index else None
    pval = float(res.pvalues[var]) if hasattr(res, 'pvalues') and var in res.pvalues.index else None

    # 95% CI (uses model's cov_params / conf_int, which should reflect HC3 if model was fit that way)
    try:
        ci_lower, ci_upper = map(float, res.conf_int(alpha=0.05).loc[var])
    except Exception:
        ci_lower = ci_upper = None

    # Because dependent variable is log(ndam15 + 1), exponentiate coef to get multiplicative effect on (ndam15+1)
    # Interpret as percent change: (exp(coef) - 1)
    exp_effect = float(np.exp(coef) - 1) if coef is not None else None
    exp_ci_lower = float(np.exp(ci_lower) - 1) if ci_lower is not None else None
    exp_ci_upper = float(np.exp(ci_upper) - 1) if ci_upper is not None else None

    # Sample size if available
    nobs = int(getattr(res, 'nobs', None)) if getattr(res, 'nobs', None) is not None else None

    # Determine statistical significance at alpha=0.05 if p-value available
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Build the returned object
    stats = {
        'variable': var,
        'coef': coef,
        'std_error': se,
        't_value': tval,
        'p_value': pval,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'exp_effect': exp_effect,            # multiplicative change in (ndam15+1) minus 1
        'exp_effect_pct': exp_effect * 100 if exp_effect is not None else None,  # percent change
        'exp_ci_pct_lower': exp_ci_lower * 100 if exp_ci_lower is not None else None,
        'exp_ci_pct_upper': exp_ci_upper * 100 if exp_ci_upper is not None else None,
        'nobs': nobs,
        'significant_at_0.05': significant
    }

    # Prepare a concise interpretation
    if coef is None:
        description = "Could not extract coefficient for FemScore_z from model output."
    else:
        direction = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)" if significant is not None else "significance unknown"
        # Interpret percent change
        if exp_effect is not None:
            pct = exp_effect * 100
            pct_lo = exp_ci_lower * 100 if exp_ci_lower is not None else None
            pct_hi = exp_ci_upper * 100 if exp_ci_upper is not None else None
            description = (
                f"FemScore_z coefficient = {coef:.4f} ({'SE=' + str(round(se,4)) if se is not None else 'SE=NA'}, "
                f"t={round(tval,2) if tval is not None else 'NA'}, p={pval:.3g} if p available). "
                f"This is {sig_text} and {direction}. "
                f"Interpreted on the original scale of log(ndam15+1): a one-standard-deviation increase in name femininity "
                f"is associated with a multiplicative change of {pct:.2f}% in (ndam15+1) "
                f"(95% CI: {pct_lo:.2f}% to {pct_hi:.2f}% if CI available). "
                f"If the coefficient is positive and statistically significant, this supports the hypothesis that more feminine names "
                f"are associated with higher fatalities (consistent with fewer precautions)."
            )
        else:
            description = (
                f"FemScore_z coefficient = {coef:.4f}. {sig_text}. "
                "Cannot compute exponentiated effect because of missing CI/coef information."
            )

    return {"object": stats, "description": description}