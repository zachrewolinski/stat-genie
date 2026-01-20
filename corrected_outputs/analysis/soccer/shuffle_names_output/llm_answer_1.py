def extract_final_answer(model_output):
    """
    Extracts the IsDark coefficient, its SE, p-value, 95% CI, and exponentiated effect (IRR)
    from a fitted statsmodels GLM/Results object (possibly cluster-robust results).
    Returns a dictionary with keys:
      - "object": dict with numeric results
      - "description": human-readable interpretation answering whether dark-skinned players
                       are more likely to receive red cards (based on p < 0.05)
    """
    import numpy as np
    import pandas as pd

    # Helper to locate the parameter name for IsDark (robust to naming differences)
    def find_param_name(res, base_name='IsDark'):
        # possible direct access
        try:
            if base_name in res.params.index:
                return base_name
        except Exception:
            pass
        # fallback: look for any param name that contains the base string (case-insensitive)
        try:
            for name in res.params.index:
                if base_name.lower() in str(name).lower():
                    return name
        except Exception:
            pass
        return None

    # Validate object has params
    if not hasattr(model_output, 'params'):
        raise ValueError("Provided model_output does not have .params attribute. "
                         "Expected a statsmodels results object.")

    param_name = find_param_name(model_output, 'IsDark')
    if param_name is None:
        raise ValueError("Could not find a parameter corresponding to 'IsDark' in the model output. "
                         "Available parameters: " + ", ".join(map(str, model_output.params.index)))

    # Extract coefficient, std err, p-value
    try:
        coef = float(model_output.params[param_name])
    except Exception as e:
        raise ValueError(f"Could not extract coefficient for {param_name}: {e}")

    # Standard error: try .bse then fallback to sqrt of diag(cov_params)
    se = None
    try:
        if hasattr(model_output, 'bse') and param_name in model_output.bse.index:
            se = float(model_output.bse[param_name])
    except Exception:
        se = None
    if se is None:
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.diag(cov))[list(cov.index).index(param_name)])
        except Exception:
            se = np.nan

    # p-value
    pvalue = np.nan
    try:
        if hasattr(model_output, 'pvalues') and param_name in model_output.pvalues.index:
            pvalue = float(model_output.pvalues[param_name])
    except Exception:
        pvalue = np.nan

    # Confidence interval: try model_output.conf_int()
    ci_lower = np.nan
    ci_upper = np.nan
    try:
        ci = model_output.conf_int()
        # conf_int may be a DataFrame or ndarray
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_lower = float(ci.loc[param_name][0])
            ci_upper = float(ci.loc[param_name][1])
        else:
            # assume ndarray with same ordering as params.index
            idx = list(model_output.params.index).index(param_name)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        # fallback using coef +- 1.96*se if se available
        if not np.isnan(se):
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

    # Exponentiate to get incidence rate ratio (IRR) because Poisson uses log link
    irr = float(np.exp(coef)) if np.isfinite(coef) else np.nan
    irr_ci_lower = float(np.exp(ci_lower)) if np.isfinite(ci_lower) else np.nan
    irr_ci_upper = float(np.exp(ci_upper)) if np.isfinite(ci_upper) else np.nan

    # Percent change interpretation
    pct_change = (irr - 1.0) * 100 if np.isfinite(irr) else np.nan
    pct_ci_lower = (irr_ci_lower - 1.0) * 100 if np.isfinite(irr_ci_lower) else np.nan
    pct_ci_upper = (irr_ci_upper - 1.0) * 100 if np.isfinite(irr_ci_upper) else np.nan

    # Try to detect covariance type used (e.g., 'cluster') to report whether SEs are cluster-robust
    cov_type = None
    try:
        cov_type = getattr(model_output, "cov_type", None)
    except Exception:
        cov_type = None

    # Formulate brief interpretation: whether effect is statistically significant at alpha=0.05
    significance = None
    if np.isfinite(pvalue):
        significance = (pvalue < 0.05)
    else:
        significance = None

    if significance is True:
        significance_statement = "The effect is statistically significant at p < 0.05."
    elif significance is False:
        significance_statement = "The effect is not statistically significant at p < 0.05."
    else:
        significance_statement = "Could not determine statistical significance (p-value unavailable)."

    # Interpretation text
    description_lines = []
    description_lines.append(f"Parameter: {param_name}")
    description_lines.append(f"Coefficient (log rate ratio): {coef:.4f}")
    description_lines.append(f"SE: {se:.4f}" if np.isfinite(se) else "SE: NA")
    description_lines.append(f"95% CI (log scale): [{ci_lower:.4f}, {ci_upper:.4f}]")
    description_lines.append(f"p-value: {pvalue:.4g}" if np.isfinite(pvalue) else "p-value: NA")
    description_lines.append(f"Incidence Rate Ratio (IRR = exp(coef)): {irr:.4f}")
    description_lines.append(f"95% CI for IRR: [{irr_ci_lower:.4f}, {irr_ci_upper:.4f}]")
    description_lines.append(f"Estimated percent change in red card rate per match for dark vs light: "
                             f"{pct_change:.1f}% (95% CI [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%])")
    description_lines.append(significance_statement)
    if cov_type is not None:
        description_lines.append(f"Reported covariance type used in results object: {cov_type}")
    description_lines.append("Because the model uses log(Matches) as an offset, these effects are multiplicative on the red-card rate per match.")

    description = " ".join(description_lines)

    result_object = {
        "param_name": str(param_name),
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "percent_change": pct_change,
        "percent_change_ci_lower": pct_ci_lower,
        "percent_change_ci_upper": pct_ci_upper,
        "cov_type": cov_type,
        "significant_at_0.05": bool(significance) if significance is not None else None
    }

    return {"object": result_object, "description": description}