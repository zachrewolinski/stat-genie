def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% confidence interval,
    and incidence-rate-ratio (IRR = exp(coef)) for the 'SkinToneDark' predictor
    from a fitted statsmodels GLM/GLMResultsWrapper (negative binomial) object.

    Returns a dictionary with keys:
      - "object": a dict with numeric results
      - "description": a human-readable interpretation string

    The function is robust to:
      - the model_output having a different exact parameter name containing 'SkinTone'
      - missing p-values (will compute from coef/SE)
      - cluster-robust results returned by get_robustcov_results
    """
    import math
    import numpy as np
    import pandas as pd

    res = model_output

    # Ensure parameter vector exists
    if not hasattr(res, 'params'):
        raise ValueError("model_output has no 'params' attribute. Provide a statsmodels results object.")

    params = res.params
    # Find the parameter name for skin tone (prefer exact 'SkinToneDark', otherwise any containing 'SkinTone')
    param_name = 'SkinToneDark'
    if param_name not in params.index:
        candidates = [n for n in params.index if 'SkinTone' in n]
        if len(candidates) == 0:
            raise KeyError("Could not find a parameter named 'SkinToneDark' or any parameter containing 'SkinTone' in model_output.params.")
        # choose the first candidate (typically there's only one)
        param_name = candidates[0]

    # Extract coefficient
    coef = float(params[param_name])

    # Standard error: try common attributes
    se = None
    if hasattr(res, 'bse') and res.bse is not None:
        se_series = res.bse
        if param_name in se_series.index:
            se = float(se_series[param_name])
    if se is None:
        # try attribute name variations
        if hasattr(res, 'std_errors') and param_name in res.std_errors.index:
            se = float(res.std_errors[param_name])
    if se is None:
        raise ValueError("Could not extract a standard error (bse) for the parameter. Ensure model_output has bse or std_errors.")

    # p-value: prefer provided pvalues, otherwise compute from z-score (Wald)
    p_value = None
    if hasattr(res, 'pvalues') and res.pvalues is not None and param_name in res.pvalues.index:
        p_value = float(res.pvalues[param_name])
    else:
        # two-sided p-value from normal distribution using z = coef / se
        z = coef / se
        # p = 2 * (1 - Phi(|z|)) where Phi is standard normal CDF
        # use erf-based CDF: Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # Confidence interval on link (log) scale: use res.conf_int()
    if hasattr(res, 'conf_int') and callable(res.conf_int):
        try:
            ci_df = res.conf_int()
            if param_name in ci_df.index:
                ci_lower = float(ci_df.loc[param_name, 0])
                ci_upper = float(ci_df.loc[param_name, 1])
            else:
                # fallback: if conf_int returns numpy array with same ordering as params
                ci_array = ci_df
                if isinstance(ci_array, (pd.DataFrame, np.ndarray)):
                    # try to align by index position
                    idx = list(params.index).index(param_name)
                    ci_lower = float(ci_array[idx, 0])
                    ci_upper = float(ci_array[idx, 1])
                else:
                    raise ValueError
        except Exception:
            # fallback to coef +/- 1.96*se
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
    else:
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Number of observations if available
    n_obs = None
    if hasattr(res, 'nobs'):
        try:
            n_obs = int(res.nobs)
        except Exception:
            n_obs = None

    # Determine simple interpretation using alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        if irr > 1.0:
            interpretation = "Statistically significant: dark-skinned players have a higher red-card rate than light-skinned players (IRR > 1)."
        else:
            interpretation = "Statistically significant: dark-skinned players have a lower red-card rate than light-skinned players (IRR < 1)."
    else:
        interpretation = "Not statistically significant at the 0.05 level: no strong evidence of a difference in red-card rates between dark- and light-skinned players."

    # Prepare return object
    result_object = {
        "parameter_name": param_name,
        "coef_log_rate": coef,                     # log rate ratio
        "std_error": se,
        "p_value": p_value,
        "ci_log_lower": ci_lower,
        "ci_log_upper": ci_upper,
        "irr": irr,                                # incidence rate ratio = exp(coef)
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "n_obs": n_obs
    }

    description = (
        f"Extracted results for '{param_name}' from Negative Binomial GLM (offset=log(Matches)). "
        f"The coefficient (log rate ratio) = {coef:.4f} (SE = {se:.4f}), 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], "
        f"p = {p_value:.4g}. Exponentiated IRR = {irr:.4f}, 95% CI = [{irr_ci_lower:.4f}, {irr_ci_upper:.4f}]. "
        f"Interpretation: IRR > 1 means dark-skinned players receive red cards at a higher rate than light-skinned players. {interpretation} "
        f"(Cluster-robust SEs should be used if model_output was produced with get_robustcov_results or similar.)"
    )

    return {"object": result_object, "description": description}