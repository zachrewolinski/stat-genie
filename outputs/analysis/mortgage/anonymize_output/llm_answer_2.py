def extract_final_answer(model_output):
    """
    Extract statistics for the 'Female' coefficient from a fitted statsmodels LogitResults object.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, z, p, CI, odds ratio, OR CI, n_obs, significance_at_0.05)
      - "description": human-readable interpretation of the Female effect in context.

    The function is defensive: it checks for None input, missing attributes, and missing parameter name.
    """
    import numpy as np

    # Basic checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model_output was provided (model_output is None)."
        }

    # Attempt to extract expected attributes from statsmodels LogitResults
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        # conf_int may be a method or attribute depending on statsmodels version
        try:
            conf = model_output.conf_int(alpha=0.05)
        except TypeError:
            conf = model_output.conf_int()  # fallback
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided model_output does not look like a fitted statsmodels results object or is missing expected attributes: {e}"
        }

    var = "Female"
    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model parameters. Available parameters: {list(params.index)}"
        }

    # Extract numeric values
    try:
        coef = float(params[var])
        se = float(bse[var])
        pval = float(pvalues[var])
        z_val = float(coef / se) if se != 0 else None

        # conf can be a DataFrame/ndarray; handle both
        try:
            ci_lower = float(conf.loc[var, 0])
            ci_upper = float(conf.loc[var, 1])
        except Exception:
            # conf is likely ndarray with rows aligned to params.index
            idx = list(params.index).index(var)
            ci_lower = float(conf[idx, 0])
            ci_upper = float(conf[idx, 1])

        or_est = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))

        # Try to get sample size
        n_obs = None
        if hasattr(model_output, "model_data") and isinstance(model_output.model_data, dict):
            n_obs = model_output.model_data.get("n_obs", None)
        if n_obs is None:
            # statsmodels stores nobs as a property/attribute
            try:
                n_obs = int(getattr(model_output, "nobs"))
            except Exception:
                # Try to infer from model endog
                try:
                    n_obs = int(model_output.model.endog.shape[0])
                except Exception:
                    n_obs = None

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting numeric statistics for '{var}': {e}"
        }

    significance = (pval < 0.05) if (pval is not None) else None

    result_object = {
        "variable": var,
        "coefficient_log_odds": coef,
        "std_error": se,
        "z_value": z_val,
        "p_value": pval,
        "95%_CI_coef": [ci_lower, ci_upper],
        "odds_ratio": or_est,
        "95%_CI_odds_ratio": [or_ci_lower, or_ci_upper],
        "n_obs": n_obs,
        "significant_at_0.05": significance
    }

    # Human-readable description
    sig_text = "statistically significant" if significance else "not statistically significant"
    description = (
        f"Estimated effect of being female on mortgage approval (controlling for listed covariates):\n"
        f"- Log-odds coefficient = {coef:.4f} (SE = {se:.4f}, z = {z_val:.2f}, p = {pval:.3g}).\n"
        f"- 95% CI for coefficient = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
        f"- Odds ratio = {or_est:.3f} with 95% CI = [{or_ci_lower:.3f}, {or_ci_upper:.3f}].\n"
        f"- Based on n = {n_obs if n_obs is not None else 'unknown'} observations. "
        f"The effect is {sig_text} at the 0.05 level.\n\n"
        f"Interpretation: holding the control variables fixed, the odds ratio describes how the odds of mortgage approval "
        f"change for female applicants compared to male applicants. An odds ratio >1 means higher odds for females, <1 means lower odds."
    )

    return {
        "object": result_object,
        "description": description
    }