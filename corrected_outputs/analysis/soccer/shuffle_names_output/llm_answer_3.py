import numpy as np

def extract_final_answer(model_output):
    """
    Extract relevant statistics for the 'SkinDark' coefficient from a fitted statsmodels results object.

    Returns a dictionary with:
      - "object": dict containing numeric results (coef, se, p-value, 95% CI, IRR, IRR 95% CI, significance flag)
      - "description": text interpretation in context

    The function is defensive to handle either DataFrame or ndarray outputs from conf_int().
    """
    res = model_output

    # Defensive extraction of core objects
    try:
        params = res.params            # pandas Series of coefficients
        pvalues = res.pvalues         # pandas Series of p-values
        bse = res.bse                 # pandas Series of standard errors
        conf_int = res.conf_int()     # DataFrame or ndarray for confidence intervals
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract parameters from model_output: {e}"
        }

    var = "SkinDark"
    try:
        param_index = list(params.index)
    except Exception:
        return {
            "object": None,
            "description": "Model parameters do not have a valid index."
        }

    if var not in param_index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model coefficients. Available variables: {param_index}"
        }

    # Extract coefficient, se, p-value
    try:
        coef = float(params[var])
    except Exception:
        coef = float(np.nan)

    try:
        se = float(bse[var]) if var in bse.index else float(np.nan)
    except Exception:
        se = float(np.nan)

    try:
        pval = float(pvalues[var]) if var in pvalues.index else float(np.nan)
    except Exception:
        pval = float(np.nan)

    # Extract 95% CI (handle DataFrame or ndarray)
    try:
        if hasattr(conf_int, "loc"):
            # pandas DataFrame/Series with index matching params
            ci_vals = conf_int.loc[var]
            # Ensure we take the first two values (low, high)
            ci_low, ci_high = float(ci_vals.iloc[0]), float(ci_vals.iloc[1])
        else:
            # conf_int is ndarray; find index of var in params.index
            idx = param_index.index(var)
            ci_low, ci_high = float(conf_int[idx, 0]), float(conf_int[idx, 1])
    except Exception:
        ci_low, ci_high = float(np.nan), float(np.nan)

    # Convert log-scale coefficient to incidence rate ratio (IRR)
    irr = float(np.exp(coef)) if np.isfinite(coef) else float(np.nan)
    irr_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else float(np.nan)
    irr_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else float(np.nan)

    signif_05 = (pval < 0.05) if (not np.isnan(pval)) else None

    result_object = {
        "variable": var,
        "coef_log_rate_ratio": coef,
        "std_error": se,
        "p_value": pval,
        "ci95_log_scale": [ci_low, ci_high],
        "IRR_rate_ratio": irr,
        "IRR_95CI": [irr_ci_low, irr_ci_high],
        "significant_p_lt_0.05": signif_05
    }

    # Short interpretation
    if signif_05 is True:
        significance_text = (
            f"The association for '{var}' is statistically significant at p = {pval:.3g}."
        )
    elif signif_05 is False:
        significance_text = (
            f"The association for '{var}' is not statistically significant (p = {pval:.3g})."
        )
    else:
        significance_text = "Could not determine statistical significance (p-value missing)."

    description = (
        f"Extracted coefficient for '{var}' from the fitted model. "
        f"Coefficient (log rate ratio) = {coef:.4f}, SE = {se:.4f}, p = {pval:.4g}, "
        f"95% CI (log scale) = [{ci_low:.4f}, {ci_high:.4f}].\n"
        f"Exponentiated effect (IRR) = {irr:.4f}, 95% CI = [{irr_ci_low:.4f}, {irr_ci_high:.4f}]. "
        f"Interpretation: IRR > 1 implies dark-skinned players receive red cards at a higher rate than light-skinned players; "
        f"IRR < 1 implies lower rate. {significance_text} "
        f"Estimates control for implicit/explicit bias scores, player goals, yellow cards, and use cluster-robust SEs by referee."
    )

    return {"object": result_object, "description": description}