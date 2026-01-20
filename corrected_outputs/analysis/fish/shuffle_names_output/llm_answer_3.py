def extract_final_answer(model_output):
    """
    Extracts rate (fish per hour) and factor effects from a fitted GLM results object
    (statsmodels.genmod.generalized_linear_model.GLMResultsWrapper).
    
    Returns a dict with:
      - "object": a dict containing:
          - "model_family": name of the fitted family (e.g., 'NegativeBinomial', 'Poisson')
          - "coef_table": pandas DataFrame with columns:
                'coef'   : estimated log-rate coefficients (log fish/hour)
                'se'     : standard errors
                'pvalue' : two-sided p-values
                'ci_low' : lower bound of 95% CI on the log scale
                'ci_high': upper bound of 95% CI on the log scale
                'rate_ratio' : exp(coef) = multiplicative effect on fish/hour
                'rr_ci_low'   : lower 95% CI for rate ratio
                'rr_ci_high'  : upper 95% CI for rate ratio
          - "baseline_rate_per_hour": exp(intercept) and its 95% CI (interpreted as expected fish/hour
                                      when predictors are at their reference/zero values)
      - "description": human-readable interpretation of the table and baseline rate.
    """
    import numpy as np
    import pandas as pd

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a statsmodels results object with .params")

    # Extract basic quantities
    params = model_output.params.copy()          # Series indexed by parameter names
    bse = model_output.bse.copy()
    pvalues = model_output.pvalues.copy()
    try:
        conf = model_output.conf_int()           # DataFrame or ndarray with index matching params
        # conf may be DataFrame; ensure columns named 0 and 1
        conf_df = pd.DataFrame(conf, index=params.index)
        conf_df.columns = ['ci_low', 'ci_high']
    except Exception:
        # Fallback: compute approximate CIs from coef +/- 1.96*se
        ci_low = params - 1.96 * bse
        ci_high = params + 1.96 * bse
        conf_df = pd.DataFrame({'ci_low': ci_low, 'ci_high': ci_high}, index=params.index)

    # Rate ratios (multiplicative effects on fish/hour)
    rate_ratio = np.exp(params)
    rr_ci_low = np.exp(conf_df['ci_low'])
    rr_ci_high = np.exp(conf_df['ci_high'])

    # Compose summary table
    summary_df = pd.DataFrame({
        'coef': params,
        'se': bse,
        'pvalue': pvalues,
        'ci_low': conf_df['ci_low'],
        'ci_high': conf_df['ci_high'],
        'rate_ratio': rate_ratio,
        'rr_ci_low': rr_ci_low,
        'rr_ci_high': rr_ci_high
    })

    # Get model family name for context
    try:
        model_family = model_output.model.family.__class__.__name__
    except Exception:
        model_family = getattr(model_output, 'family', None) or 'Unknown'

    # Baseline rate: exp(intercept). Intercept should be named 'const' since sm.add_constant was used.
    intercept_name = None
    for possible in ['const', 'Intercept', 'intercept', 'CONST']:
        if possible in params.index:
            intercept_name = possible
            break

    if intercept_name is None:
        # If no explicit intercept found, try the first index entry (less ideal)
        intercept_name = params.index[0]

    intercept = params.loc[intercept_name]
    intercept_ci_low = conf_df.loc[intercept_name, 'ci_low']
    intercept_ci_high = conf_df.loc[intercept_name, 'ci_high']

    baseline_rate = float(np.exp(intercept))              # expected fish per hour at reference
    baseline_rate_ci = (float(np.exp(intercept_ci_low)), float(np.exp(intercept_ci_high)))

    result_object = {
        "model_family": model_family,
        "coef_table": summary_df,
        "baseline_rate_per_hour": {
            "value": baseline_rate,
            "95%_ci": baseline_rate_ci,
            "interpretion": (
                "Expected number of fish caught per hour when predictors are at their reference/zero "
                "(e.g., livebait=0, persons_z=0 (average group size), camper_z=0 (average campers), child=0)."
            )
        }
    }

    description = (
        "This output gives the fitted log-rate coefficients from a GLM with log link and offset=log_hours, "
        "so coefficients are on the log(fish/hour) scale. Exponentiating a coefficient gives a rate ratio (multiplicative "
        "change in fish/hour) for a one-unit increase in the predictor. For binary predictors (e.g., livebait, child), the "
        "rate ratio compares the group with the feature (1) to the reference (0). For standardized continuous predictors "
        "(persons_z, camper_z), the rate ratio corresponds to a one-standard-deviation change.\n\n"
        "Key elements returned:\n"
        "- coef_table: table with coef, se, pvalue, 95% CI (on log scale), rate_ratio = exp(coef), and 95% CI for the rate ratio.\n"
        "- baseline_rate_per_hour: exp(intercept) and its 95% CI, interpreted as expected fish/hour when predictors are at reference.\n\n"
        "Interpretation example (how to read results):\n"
        "- If coef for 'livebait' = 0.30 (rate_ratio ≈ 1.35) and p < 0.05, using live bait is associated with ~35% higher fish/hour.\n"
        "- If persons_z rate_ratio = 1.10, then a one-standard-deviation larger group is associated with ~10% higher fish/hour.\n\n"
        "Use the returned 'coef_table' to see which predictors have statistically significant effects (pvalue) and their "
        "magnitude (rate_ratio). The baseline_rate_per_hour gives an estimate of average fish caught per hour for the reference profile."
    )

    return {"object": result_object, "description": description}