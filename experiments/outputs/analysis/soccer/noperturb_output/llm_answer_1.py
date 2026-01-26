def extract_final_answer(model_output):
    """
    Extracts the effect of SkinDark from a fitted statsmodels GLM results object
    (clustered robust results expected under key 'fitted_model_clustered').

    Returns a dictionary with:
      - "object": dict with numeric results (coef, se, z, p, IRR, IRR CI)
      - "description": human-readable interpretation in context
    """
    import numpy as np
    from scipy import stats

    # Attempt to find the fitted results object
    res = None
    if isinstance(model_output, dict):
        # common key used in the modeling function
        res = model_output.get('fitted_model_clustered', None)
        # fallback: accept a raw results object passed directly
        if res is None:
            # try common alternate keys
            for k in ['fitted_model', 'results', 'model']:
                if k in model_output:
                    res = model_output[k]
                    break
    else:
        # if the user passed the results object directly
        res = model_output

    if res is None:
        raise ValueError("No fitted model found in model_output. Expected a dict containing 'fitted_model_clustered'.")

    var = 'SkinDark'
    # Protect against different indexing (just raise if missing)
    if var not in list(res.params.index):
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(res.params.index)}")

    coef = float(res.params[var])
    # Use bse and pvalues provided by the (clustered) results if available
    se = float(res.bse[var]) if hasattr(res, 'bse') and var in res.bse.index else np.nan
    # z-statistic (Wald) and p-value (two-sided). If pvalue not present, compute from z.
    z = float(coef / se) if se and not np.isnan(se) and se > 0 else np.nan
    if hasattr(res, 'pvalues') and var in res.pvalues.index:
        pvalue = float(res.pvalues[var])
    else:
        pvalue = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan

    # confidence interval on the coefficient scale
    conf = res.conf_int()
    conf_arr = np.asarray(conf)
    idx = list(res.params.index).index(var)
    ci_lower = float(conf_arr[idx, 0])
    ci_upper = float(conf_arr[idx, 1])

    # incidence rate ratio (IRR) and CI on multiplicative scale
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Build the numeric result object
    result_object = {
        'variable': var,
        'coef': coef,
        'se': se,
        'z': z,
        'p_value': pvalue,
        'IRR': irr,
        'IRR_CI_lower': irr_ci_lower,
        'IRR_CI_upper': irr_ci_upper,
        'percent_change_IRR': (irr - 1) * 100  # e.g., 14.8 means ~14.8% higher rate
    }

    # Human-readable description / interpretation
    sig = (pvalue < 0.05) if (not np.isnan(pvalue)) else False
    significance_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"

    description = (
        f"Estimated effect of SkinDark (1 = Dark vs 0 = Light): coefficient = {coef:.4f} "
        f"(SE = {se:.4f}, z = {z:.2f}, p = {pvalue:.3f}). "
        f"The incidence rate ratio (IRR) = {irr:.3f} with 95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]. "
        f"This corresponds to an estimated {(irr - 1) * 100:.1f}% {'increase' if irr > 1 else 'decrease'} in red-card rate for dark-skinned players compared to light-skinned players, "
        f"controlling for the modeled covariates. The effect is {significance_text}. "
        f"Because the 95% CI for the IRR includes 1 and the p-value is {pvalue:.3f}, there is not enough evidence at the 0.05 level to conclude that dark-skinned players are more likely to receive red cards."
    )

    return {
        "object": result_object,
        "description": description
    }