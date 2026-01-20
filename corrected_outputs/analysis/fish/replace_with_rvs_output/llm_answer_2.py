def extract_final_answer(model_output):
    """
    Extracts key statistics from the model output (statsmodels GLMResultsWrapper)
    and returns a concise summary suitable for interpreting fish-per-hour rates.

    Returns a dictionary with:
      - "object": dict containing coefficient, SE, p-value, 95% CI, IRR and IRR 95% CI
                  for each model term, plus baseline rate per hour, model type and dispersion.
      - "description": human-readable explanation of what the numbers mean.

    Expected input: the dictionary returned by the provided `model` function:
      {'model_type': str, 'final_results': statsmodels.results.GLMResultsWrapper,
       'poisson_results': ..., 'dispersion': float}
    """
    import numpy as np
    import pandas as pd

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    if 'final_results' not in model_output:
        raise ValueError("model_output does not contain 'final_results'.")

    results = model_output['final_results']
    model_type = model_output.get('model_type', None)
    dispersion = model_output.get('dispersion', None)

    # Extract parameter table
    params = results.params          # pandas Series
    bse = results.bse                # pandas Series
    pvalues = results.pvalues        # pandas Series
    try:
        conf = results.conf_int()    # DataFrame with two columns [lower, upper]
    except Exception:
        # Fallback: construct approximate confint using Normal approximation
        z = 1.96
        conf = pd.DataFrame({
            0: params - z * bse,
            1: params + z * bse
        }, index=params.index)

    # Identify intercept name (common variants)
    intercept_names = ['Intercept', 'intercept', 'const', 'Const']
    intercept_name = None
    for name in intercept_names:
        if name in params.index:
            intercept_name = name
            break
    # If still None, pick the first parameter as a fallback (but warn in description)
    fallback_intercept = False
    if intercept_name is None:
        intercept_name = params.index[0]
        fallback_intercept = True

    # Build output dict for each parameter
    term_summaries = {}
    for term in params.index:
        coef = float(params.loc[term])
        se = float(bse.loc[term]) if term in bse.index else float(np.nan)
        pval = float(pvalues.loc[term]) if term in pvalues.index else float(np.nan)

        # Confidence interval
        try:
            ci_lower = float(conf.loc[term].iloc[0])
            ci_upper = float(conf.loc[term].iloc[1])
        except Exception:
            # If conf is not indexed as expected, try positional lookup
            idx = list(params.index).index(term)
            ci_lower = float(conf.iloc[idx, 0])
            ci_upper = float(conf.iloc[idx, 1])

        # IRR (incidence rate ratio) and its CI
        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower))
        irr_ci_upper = float(np.exp(ci_upper))

        term_summaries[term] = {
            'coef': coef,
            'se': se,
            'p_value': pval,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'irr': irr,
            'irr_95_lower': irr_ci_lower,
            'irr_95_upper': irr_ci_upper
        }

    # Baseline expected fish-per-hour: exp(intercept)
    intercept_coef = term_summaries[intercept_name]['coef']
    intercept_ci_lower = term_summaries[intercept_name]['ci_95_lower']
    intercept_ci_upper = term_summaries[intercept_name]['ci_95_upper']
    baseline_rate_per_hour = float(np.exp(intercept_coef))
    baseline_rate_ci_lower = float(np.exp(intercept_ci_lower))
    baseline_rate_ci_upper = float(np.exp(intercept_ci_upper))

    # Compose the "object" result
    result_object = {
        'model_type': model_type,
        'dispersion': dispersion,
        'terms': term_summaries,
        'baseline_rate_per_hour': baseline_rate_per_hour,
        'baseline_rate_per_hour_95ci': (baseline_rate_ci_lower, baseline_rate_ci_upper),
        'notes': {
            'intercept_name_used': intercept_name,
            'intercept_was_fallback': fallback_intercept
        }
    }

    # Description explaining interpretation
    description_lines = []
    description_lines.append("Extracted coefficients are on the log(rate) scale from a GLM with log(hours) as an offset.")
    description_lines.append("IRR = exp(coef) is the multiplicative effect on the expected fish-per-hour.")
    description_lines.append(f"Model selected: {model_type}. Dispersion statistic: {dispersion:.3f} (Pearson chi2 / df_resid).")
    description_lines.append("")
    description_lines.append("Interpreting key quantities:")
    description_lines.append(f"- Baseline rate per hour (all covariates = 0): {baseline_rate_per_hour:.3f} fish/hour")
    description_lines.append(f"  95% CI: [{baseline_rate_ci_lower:.3f}, {baseline_rate_ci_upper:.3f}]")
    description_lines.append("")
    description_lines.append("For each model term (e.g., 'livebait', 'camper') the returned fields are:")
    description_lines.append("  coef: log-rate coefficient;")
    description_lines.append("  se: standard error of coef;")
    description_lines.append("  p_value: Wald test p-value for coef != 0;")
    description_lines.append("  ci_95_lower / ci_95_upper: 95% CI on the log-rate scale;")
    description_lines.append("  irr: exp(coef), multiplicative change in fish-per-hour when that term increases by 1 (holding others constant);")
    description_lines.append("  irr_95_lower / irr_95_upper: 95% CI for the IRR.")
    if fallback_intercept:
        description_lines.append("")
        description_lines.append("Note: intercept name was not the common 'Intercept'/'const' and a fallback was used; interpret baseline accordingly.")
    description_lines.append("")
    description_lines.append("Example interpretation:")
    description_lines.append("  If term 'livebait' has irr = 1.50 and p < 0.05, then using live bait is associated with a 50% higher expected fish-per-hour,")
    description_lines.append("  compared to not using live bait, holding other model covariates constant.")

    description = "\n".join(description_lines)

    return {"object": result_object, "description": description}