def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted model output and return a concise
    summary suitable for interpreting fish caught per hour.

    Returns a dictionary with keys:
      - "object": a dict containing the model used, baseline rate (fish/hour),
                  a table-like list of coefficient summaries (coef, SE, p, IRR,
                  95% CI on IRR), and a short numeric interpretation of the
                  livebait effect (if present).
      - "description": a short text explaining what the returned numbers mean.

    Expects model_output to be the dict returned by the provided `model` function,
    i.e. containing keys 'poisson', 'neg_bin', 'dispersion', 'use_negative_binomial'.
    """
    import numpy as np
    import pandas as pd

    # Determine which model to use (prefer Negative Binomial if indicated)
    use_nb = bool(model_output.get('use_negative_binomial', False))
    model_key = 'neg_bin' if use_nb and ('neg_bin' in model_output) else 'poisson'
    res = model_output.get(model_key, None)
    if res is None:
        raise ValueError(f"Requested model '{model_key}' not found in model_output")

    # Extract parameters, standard errors, p-values, and confidence intervals
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    # conf_int returns a DataFrame-like with two columns (lower, upper)
    try:
        ci = res.conf_int()
    except Exception:
        # fallback: approximate CI using normal approx
        ci = pd.DataFrame({
            0: params - 1.96 * bse,
            1: params + 1.96 * bse
        }, index=params.index)

    # Build coefficient summary table with exponentiated effects (IRR)
    coef_rows = []
    for name in params.index:
        coef = float(params[name])
        se = float(bse.get(name, np.nan))
        p = float(pvalues.get(name, np.nan))
        ci_low = float(ci.loc[name, 0])
        ci_high = float(ci.loc[name, 1])
        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low))
        irr_ci_high = float(np.exp(ci_high))

        coef_rows.append({
            'term': name,
            'coef_log': coef,
            'se': se,
            'p_value': p,
            'IRR': irr,
            'IRR_95CI_lower': irr_ci_low,
            'IRR_95CI_upper': irr_ci_high
        })

    # Estimate baseline expected fish per hour:
    # The intercept corresponds to the log(rate) when all predictors are at their reference/centered values.
    intercept_name = None
    for n in params.index:
        if n.lower() in ('intercept', 'const'):
            intercept_name = n
            break
    # Patsy/statsmodels often names the intercept "Intercept"
    if intercept_name is None and 'Intercept' in params.index:
        intercept_name = 'Intercept'

    if intercept_name is not None:
        baseline_log_rate = float(params[intercept_name])
        baseline_rate_per_hour = float(np.exp(baseline_log_rate))
    else:
        baseline_rate_per_hour = None  # cannot compute

    # Provide a direct numeric interpretation of the livebait effect if present
    livebait_info = None
    if 'livebait' in params.index:
        row = next(r for r in coef_rows if r['term'] == 'livebait')
        livebait_info = {
            'term': 'livebait',
            'IRR': row['IRR'],
            'IRR_95CI': (row['IRR_95CI_lower'], row['IRR_95CI_upper']),
            'p_value': row['p_value'],
            'interpretation': (
                f"Using live bait multiplies the expected fish-per-hour by {row['IRR']:.3f} "
                f"(95% CI: {row['IRR_95CI_lower']:.3f}–{row['IRR_95CI_upper']:.3f}). "
                f"If p < 0.05 this effect is statistically significant (p={row['p_value']:.3g})."
            )
        }

    # Compose return object
    result_object = {
        'model_used': model_key,
        'dispersion': float(model_output.get('dispersion', np.nan)),
        'baseline_rate_per_hour': (float(baseline_rate_per_hour) if baseline_rate_per_hour is not None else None),
        'coefficients': coef_rows,
        'livebait_summary': livebait_info
    }

    # Short human-readable description
    description_lines = [
        f"Model used: {model_key} (Negative Binomial preferred due to dispersion={result_object['dispersion']:.3g}).",
    ]
    if baseline_rate_per_hour is not None:
        description_lines.append(
            f"Baseline expected catch rate = {baseline_rate_per_hour:.3f} fish per hour when all predictors are at their reference/centered values "
            "(reference county, mean-centered covariates)."
        )
    else:
        description_lines.append("Baseline expected catch rate could not be computed (no intercept found).")

    description_lines.append(
        "For each model term we return the log-coefficient, robust SE, p-value, the incidence-rate-ratio (IRR = exp(coef)), "
        "and the 95% CI for the IRR. For example, an IRR of 1.5 means a 50% higher expected fish/hour for a one-unit increase in that predictor."
    )
    if livebait_info is not None:
        description_lines.append("A quick summary for livebait is provided under 'livebait_summary' in the returned object.")

    description = " ".join(description_lines)

    return {'object': result_object, 'description': description}