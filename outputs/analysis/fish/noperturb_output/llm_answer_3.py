def extract_final_answer(model_output):
    """
    Extract interpretable statistics from the fitted model output and return:
      - "object": a pandas DataFrame with coefficients, SEs, p-values, IRRs and 95% CI for IRRs
      - "description": a short human-readable interpretation of the key results
    
    Expected input: the dictionary returned by the modeling function (contains
    'poisson_results', 'nb_results', 'use_negative_binomial', 'dispersion', etc.)
    """
    import numpy as np
    import pandas as pd

    # Determine which fitted model to use (prefer Negative Binomial if used and available)
    use_nb = bool(model_output.get('use_negative_binomial', False))
    nb_res = model_output.get('nb_results', None)
    poisson_res = model_output.get('poisson_results', None)

    if use_nb and nb_res is not None:
        model_res = nb_res
        model_name = 'Negative Binomial'
    elif poisson_res is not None:
        model_res = poisson_res
        model_name = 'Poisson'
    else:
        raise ValueError("No fitted model results found in model_output.")

    # Extract coefficient table
    params = model_res.params
    try:
        bse = model_res.bse
    except Exception:
        # fallback if bse attribute missing
        bse = model_res.bsevalues if hasattr(model_res, 'bsevalues') else np.nan * params

    pvalues = model_res.pvalues
    try:
        ci = model_res.conf_int()
    except Exception:
        # If conf_int fails, approximate using normal approx
        z = 1.96
        ci_lower = params - z * bse
        ci_upper = params + z * bse
        ci = pd.DataFrame({'lower': ci_lower, 'upper': ci_upper}, index=params.index)
    # Ensure ci is a DataFrame with two columns
    if isinstance(ci, np.ndarray):
        ci = pd.DataFrame(ci, index=params.index)
    if ci.shape[1] == 2:
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
    else:
        # fallback
        ci_lower = params - 1.96 * bse
        ci_upper = params + 1.96 * bse

    # Compute incidence rate ratios (IRR) and their CIs
    irr = np.exp(params)
    irr_ci_lower = np.exp(ci_lower)
    irr_ci_upper = np.exp(ci_upper)

    # Build result DataFrame
    result_df = pd.DataFrame({
        'coefficient': params,
        'std_err': bse,
        'p_value': pvalues,
        'IRR': irr,
        'IRR_CI_lower': irr_ci_lower,
        'IRR_CI_upper': irr_ci_upper
    })

    # Round for neatness (but keep numeric types)
    result_df = result_df.round({
        'coefficient': 4,
        'std_err': 4,
        'p_value': 4,
        'IRR': 4,
        'IRR_CI_lower': 4,
        'IRR_CI_upper': 4
    })

    # Identify statistically significant predictors (alpha = 0.05)
    sig_mask = result_df['p_value'] < 0.05
    significant = result_df[sig_mask]

    # Prepare textual summary
    lines = []
    lines.append(f"Model used: {model_name}.")
    dispersion = model_output.get('dispersion', None)
    if dispersion is not None:
        lines.append(f"Poisson dispersion statistic (from initial fit): {dispersion:.3f}.")
    lines.append("Returned table contains: coefficient (log-rate), std_err, p_value, IRR (exp(coef)), and 95% CI for IRR.")
    lines.append("Interpretation of IRR: multiplicative change in expected fish caught per hour for a one-unit increase in the predictor (or compared to reference for binary predictors).")
    if not significant.empty:
        lines.append("Statistically significant predictors (p < 0.05):")
        for name, row in significant.iterrows():
            lines.append(
                f" - {name}: IRR = {row['IRR']:.3f} (95% CI [{row['IRR_CI_lower']:.3f}, {row['IRR_CI_upper']:.3f}]), p = {row['p_value']:.3f}"
            )
    else:
        lines.append("No predictors reach conventional statistical significance at p < 0.05.")

    # Give short guidance on interpretation for variable types
    lines.append("Example interpretations:")
    lines.append(" - For a binary variable (e.g., livebait=1 vs 0): IRR > 1 means groups using live bait catch more fish per hour (multiplicative factor = IRR).")
    lines.append(" - For a continuous variable (e.g., group_size): IRR represents the multiplicative change in catch rate per one-unit increase in group_size.")
    description = " ".join(lines)

    return {"object": result_df, "description": description}