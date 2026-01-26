def extract_final_answer(model_output):
    """
    Extracts statistics about the 'IsHuman' effect from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with:
      - "object": dict of numeric results (coef, se, p, 95% CI, odds ratio and its CI, predicted probability difference if available, significance flag)
      - "description": brief plain-language conclusion about whether modern humans have higher AMTL
    
    Expects the model to contain a parameter whose name includes 'IsHuman' (e.g., 'IsHuman').
    """
    import numpy as np
    import math

    # Helper logistic function
    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    res = model_output  # alias

    # Gather parameter names and try to find the IsHuman parameter
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()  # DataFrame-like or array
    except Exception as e:
        raise ValueError(f"Provided object does not look like a statsmodels results wrapper: {e}")

    # Identify the parameter corresponding to IsHuman
    param_name = None
    # direct name
    if 'IsHuman' in params.index:
        param_name = 'IsHuman'
    else:
        # any parameter name containing the substring 'IsHuman'
        for n in params.index:
            if 'IsHuman' in str(n):
                param_name = n
                break

    if param_name is None:
        raise ValueError("Could not find a parameter name containing 'IsHuman' in the model parameters index.")

    # Extract statistics
    coef = float(params[param_name])
    se = float(bse[param_name]) if param_name in bse.index else float(bse[params.index.get_loc(param_name)])
    pval = float(pvalues[param_name])
    # conf could be a DataFrame with index matching params.index, or an ndarray in the same order
    try:
        ci_lower, ci_upper = conf.loc[param_name].astype(float)
    except Exception:
        # fallback: conf as ndarray, find index
        idx = list(params.index).index(param_name)
        ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])

    # Odds ratio and CI
    or_val = math.exp(coef)
    or_ci_lower = math.exp(ci_lower)
    or_ci_upper = math.exp(ci_upper)

    # Determine significance at alpha=0.05
    significant = (pval < 0.05)

    # Try to compute an approximate average predicted probability difference
    prob_diff = None
    prob0 = None
    prob1 = None
    try:
        # Access model exog and names
        exog = res.model.exog.copy()  # (n_obs, n_params)
        exog_names = res.model.exog_names
        if isinstance(exog_names, (list, tuple)):
            if any('IsHuman' in str(n) for n in exog_names):
                is_idx = next(i for i, n in enumerate(exog_names) if 'IsHuman' in str(n))
                # create copies with IsHuman column set to 0 or 1
                exog0 = exog.copy()
                exog1 = exog.copy()
                exog0[:, is_idx] = 0.0
                exog1[:, is_idx] = 1.0
                # compute mean linear predictor for each scenario
                lp0 = np.mean(np.dot(exog0, params))
                lp1 = np.mean(np.dot(exog1, params))
                prob0 = float(logistic(lp0))
                prob1 = float(logistic(lp1))
                prob_diff = float(prob1 - prob0)
    except Exception:
        # If any step fails, we will leave prob_diff as None (optional info)
        prob_diff = None

    # Prepare numeric object to return
    numeric_result = {
        "param_name": str(param_name),
        "coef": coef,
        "std_error": se,
        "p_value": pval,
        "95%_CI_coef": [ci_lower, ci_upper],
        "odds_ratio": or_val,
        "95%_CI_odds_ratio": [or_ci_lower, or_ci_upper],
        "significant_at_0.05": bool(significant),
        "predicted_prob_mean_IsHuman0": prob0,
        "predicted_prob_mean_IsHuman1": prob1,
        "predicted_prob_difference": prob_diff
    }

    # Formulate brief description / conclusion
    if significant:
        if coef > 0:
            conclusion = (
                f"The model coefficient for '{param_name}' is positive (coef = {coef:.4f}, "
                f"OR = {or_val:.3f}, 95% CI OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}], p = {pval:.3g}). "
                "This indicates that, after controlling for age, sex, and tooth class, modern humans "
                "have statistically significantly higher odds of AMTL than the non-human primate genera in the sample."
            )
        else:
            conclusion = (
                f"The model coefficient for '{param_name}' is negative (coef = {coef:.4f}, "
                f"OR = {or_val:.3f}, 95% CI OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}], p = {pval:.3g}). "
                "This indicates that, after controlling for age, sex, and tooth class, modern humans "
                "have statistically significantly lower odds of AMTL than the non-human primate genera in the sample."
            )
    else:
        conclusion = (
            f"The model coefficient for '{param_name}' is {coef:.4f} (OR = {or_val:.3f}, "
            f"95% CI OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}], p = {pval:.3g}). "
            "This effect is not statistically significant at the 0.05 level, so we do not have evidence "
            "that modern humans differ from the sampled non-human primate genera in AMTL after controlling "
            "for age, sex, and tooth class."
        )

    # Add an optional note about predicted probability difference if available
    if prob_diff is not None:
        conclusion += (
            f" Approximate average predicted AMTL proportion when toggling IsHuman from 0→1 is "
            f"{prob_diff:.4f} (mean predicted AMTL: IsHuman=0 → {prob0:.4f}; IsHuman=1 → {prob1:.4f})."
        )

    return {
        "object": numeric_result,
        "description": conclusion
    }