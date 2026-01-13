def extract_final_answer(model_output):
    """
    Extracts the statistics for the IsHuman predictor from a fitted statsmodels GLMResults-like object.

    Returns a dictionary with:
      - "object": dict with numeric results for the IsHuman coefficient (log-odds), SE, z/t, p-value,
                  95% CI (log-odds), odds ratio and its 95% CI, and a boolean 'significant' (p < 0.05).
      - "description": short textual interpretation of these statistics in the context of the task.

    If the IsHuman parameter cannot be found in the model, the function returns an explanation.
    """
    import re
    import numpy as np

    # Validate model_output has the attributes we need
    required_attrs = ['params', 'bse', 'tvalues', 'pvalues', 'conf_int']
    for attr in required_attrs:
        if not hasattr(model_output, attr):
            return {
                "object": None,
                "description": f"Input model_output does not have required attribute '{attr}'."
            }

    # Find the parameter name corresponding to IsHuman
    param_names = list(model_output.params.index)
    ishuman_candidates = [n for n in param_names if re.search(r'IsHuman', n)]
    if len(ishuman_candidates) == 0:
        return {
            "object": None,
            "description": "Could not find a parameter name containing 'IsHuman' in the model's parameters."
        }
    # choose the first matching parameter name
    pname = ishuman_candidates[0]

    # Extract statistics
    try:
        coef = float(model_output.params[pname])
        se = float(model_output.bse[pname])
        # statsmodels GLM uses .tvalues for z/t; also some versions provide .tvalues or .zvalues
        # we'll try tvalues then fall back to zvalues if present
        if hasattr(model_output, 'tvalues'):
            stat = float(model_output.tvalues[pname])
        elif hasattr(model_output, 'zvalues'):
            stat = float(model_output.zvalues[pname])
        else:
            stat = None
        pval = float(model_output.pvalues[pname]) if pname in model_output.pvalues.index else None

        ci_df = model_output.conf_int()
        if pname in ci_df.index:
            ci_low = float(ci_df.loc[pname, 0])
            ci_high = float(ci_df.loc[pname, 1])
        else:
            ci_low, ci_high = None, None

        # Odds ratio and CI on odds ratio scale
        or_coef = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        significant = (pval is not None) and (pval < 0.05)

        result_obj = {
            "predictor": pname,
            "coef_log_odds": coef,
            "se": se,
            "z_or_t": stat,
            "p_value": pval,
            "ci_log_odds_95%": [ci_low, ci_high],
            "odds_ratio": or_coef,
            "ci_odds_ratio_95%": [or_ci_low, or_ci_high],
            "significant_at_0.05": significant
        }

        # Build human-readable interpretation
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        p_sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p ≥ 0.05)"
        description = (
            f"Parameter '{pname}' estimates the difference in log-odds of antemortem tooth loss (AMTL) for modern humans "
            f"relative to non-human primates, controlling for age, prob_male, and tooth_class. "
            f"The estimated coefficient is {coef:.4f} (SE = {se:.4f}), z/t = {stat:.3f}, p = {pval:.4g}. "
            f"The 95% CI for the log-odds is [{ci_low:.4f}, {ci_high:.4f}]. "
            f"This corresponds to an odds ratio of {or_coef:.3f} with 95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}]. "
            f"Because the coefficient is {direction} and the result is {p_sig_text}, "
            f"we {'have' if significant else 'do not have'} evidence that modern humans have {'higher' if coef>0 else 'lower' if coef<0 else 'different'} "
            f"AMTL frequency compared to the non-human primate genera after accounting for the covariates."
        )

        return {"object": result_obj, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting statistics for parameter '{pname}': {e}"
        }