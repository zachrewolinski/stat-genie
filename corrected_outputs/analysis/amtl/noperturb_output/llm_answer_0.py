def extract_final_answer(model_output):
    """
    Extract statistics for the 'IsHuman' effect from a fitted statsmodels GLMResults-like object
    (expected to already have clustered-robust covariance if applicable).

    Returns:
      {
        "object": { ... detailed stats ... },
        "description": "Plain-language summary and interpretation"
      }

    The function is defensive about the exact parameter name for the IsHuman variable:
    it tries to find a parameter name that contains 'IsHuman'. If not found it will raise
    a ValueError.
    """
    import numpy as np

    # Ensure model_output looks like a statsmodels results object
    if model_output is None:
        raise ValueError("model_output is None")

    # Access parameters, standard errors, p-values, and conf_int
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf = model_output.conf_int()  # DataFrame/array with lower, upper
    except Exception as e:
        raise ValueError(f"Provided model_output does not expose expected attributes: {e}")

    # Find the parameter name for IsHuman (be permissive: look for substring)
    ishuman_names = [n for n in params.index if 'IsHuman' in str(n)]
    if len(ishuman_names) == 0:
        # Try other common variants
        alt_names = [n for n in params.index if 'human' in str(n).lower()]
        if len(alt_names) == 0:
            raise ValueError(f"Could not find a parameter name containing 'IsHuman' or 'human'. "
                             f"Available parameter names: {list(params.index)}")
        ishuman_names = alt_names

    # If multiple matches, prefer exact 'IsHuman' else take first
    if 'IsHuman' in ishuman_names:
        pname = 'IsHuman'
    else:
        pname = ishuman_names[0]

    coef = float(params[pname])
    se = float(bse[pname]) if pname in bse.index else float(np.nan)
    pval = float(pvalues[pname]) if pname in pvalues.index else float(np.nan)

    # z-statistic (Wald z using robust SEs)
    z_stat = coef / se if se != 0 else float('nan')

    # Odds ratio and 95% CI on OR scale
    try:
        ci_lower, ci_upper = conf.loc[pname].iloc[0], conf.loc[pname].iloc[1]
    except Exception:
        # conf may be an array-like
        try:
            row = list(conf.index).index(pname)
            ci_lower, ci_upper = conf[row, 0], conf[row, 1]
        except Exception:
            ci_lower, ci_upper = float('nan'), float('nan')

    or_point = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else float('nan')
    or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else float('nan')

    # Determine a simple yes/no answer about whether humans have higher AMTL,
    # using positive coef + p < 0.05 as evidence for "higher".
    alpha = 0.05
    humans_higher = (coef > 0) and (pval < alpha)

    # Construct object to return with the key statistics
    result_object = {
        "parameter_name": pname,
        "coef_logit": coef,
        "std_error": se,
        "z_stat": z_stat,
        "p_value": pval,
        "odds_ratio": or_point,
        "odds_ratio_95CI": [or_ci_lower, or_ci_upper],
        "humans_higher_significant_at_0.05": bool(humans_higher),
        "note": "Model is a binomial GLM (logit link). Coef is log-odds; odds ratio = exp(coef). "
                "Standard errors, p-values, and CIs are taken from the provided results object (assumed clustered-robust if returned by get_robustcov_results)."
    }

    # Plain-language description
    if np.isnan(pval):
        desc = (f"Could not determine statistical evidence for 'IsHuman' because p-value is not available. "
                f"Parameter '{pname}' has logit coef={coef:.4f}.")
    else:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        sig_text = "statistically significant" if pval < alpha else "not statistically significant"
        desc = (
            f"The model parameter '{pname}' has a log-odds coefficient = {coef:.4f} (SE = {se:.4f}, z = {z_stat:.3f}, p = {pval:.3g}). "
            f"Exponentiated, this is an odds ratio = {or_point:.3f} with 95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]. "
            f"This indicates that, controlling for age, sex (prob_male), and tooth class, modern humans have {direction} AMTL risk compared to the reference non-human primates. "
            f"The effect is {sig_text} at alpha = {alpha}."
        )

    return {"object": result_object, "description": desc}