def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, and odds-ratios
    for predictors relevant to the question:
      - relative_size_ratio (primary predictor)
      - adv_home (continuous location / home-range advantage)
      - size_adv_interaction (interaction, if present)
      - Location_FocalHome (dummy)
      - Location_OtherHome (dummy)
    Returns a dictionary with:
      - "object": dict mapping each variable -> extracted stats (or None if missing)
      - "description": short interpretation of the results and how to read them
    """
    import numpy as np

    res = model_output

    # Try common attribute access patterns for statsmodels results (robust or not)
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
    except Exception:
        # If not present, try underlying results attribute
        try:
            params = res._results.params
            bse = res._results.bse
            pvalues = res._results.pvalues
        except Exception:
            raise ValueError("Could not find params/bse/pvalues on the provided model_output object.")

    # Try to get confidence intervals; if not available compute approx using normal approx
    try:
        ci = res.conf_int()
        # conf_int returns array-like with two columns [lower, upper]
    except Exception:
        # compute approximate 95% CI: param +/- 1.96 * se
        lower = params - 1.96 * bse
        upper = params + 1.96 * bse
        ci = np.column_stack([lower, upper])

    # Ensure we can index arrays by parameter names
    # Convert params, bse, pvalues to pandas Series-like if they are numpy arrays with index in res.model.exog_names
    try:
        param_index = params.index
    except Exception:
        # attempt to get names from model
        try:
            param_index = res.model.exog_names
            # convert to Series for easier indexing
            import pandas as pd
            params = pd.Series(params, index=param_index)
            bse = pd.Series(bse, index=param_index)
            pvalues = pd.Series(pvalues, index=param_index)
            ci = pd.DataFrame(ci, index=param_index, columns=["2.5%", "97.5%"])
        except Exception:
            # fallback: treat as unnamed numeric arrays
            param_index = None

    # If ci is a numpy array without index, try to align using param_index
    import pandas as pd
    if not isinstance(ci, (pd.DataFrame, pd.Series)):
        try:
            ci = pd.DataFrame(ci, index=param_index, columns=["2.5%", "97.5%"])
        except Exception:
            # if all else fails, create DataFrame with numeric indices
            ci = pd.DataFrame(ci, columns=["2.5%", "97.5%"])

    # Variables of interest
    vars_of_interest = [
        'relative_size_ratio',
        'adv_home',
        'size_adv_interaction',
        'Location_FocalHome',
        'Location_OtherHome'
    ]

    extracted = {}
    for v in vars_of_interest:
        if (param_index is not None and v in param_index) or (hasattr(params, "__contains__") and v in params):
            coef = float(params[v])
            se = float(bse[v])
            pval = float(pvalues[v])
            ci_low = float(ci.loc[v].iloc[0]) if v in ci.index else float(ci.iloc[params.index.get_loc(v), 0])
            ci_high = float(ci.loc[v].iloc[1]) if v in ci.index else float(ci.iloc[params.index.get_loc(v), 1])
            odds_ratio = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))

            extracted[v] = {
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_2.5%": or_ci_low,
                "odds_ratio_ci_97.5%": or_ci_high
            }
        else:
            extracted[v] = None

    # Optionally extract some model-level summaries if available
    model_info = {}
    try:
        model_info['n_obs'] = int(res.nobs)
    except Exception:
        try:
            model_info['n_obs'] = int(res.model.endog.shape[0])
        except Exception:
            model_info['n_obs'] = None
    try:
        model_info['pseudo_R2'] = float(res.prsquared)
    except Exception:
        model_info['pseudo_R2'] = None
    try:
        model_info['llf'] = float(res.llf)
        model_info['llnull'] = float(res.llnull)
    except Exception:
        pass

    result_obj = {
        "predictor_stats": extracted,
        "model_info": model_info
    }

    description_lines = [
        "For each predictor, 'coef' is the logistic regression coefficient (change in log-odds of the focal group winning per unit increase in the predictor).",
        "'odds_ratio' = exp(coef) gives the multiplicative change in the odds of the focal group winning per unit increase.",
        "'ci_2.5%' and 'ci_97.5%' are the 95% confidence interval bounds on the coefficient (log-odds scale); the exponentiated CI bounds show the 95% CI for the odds ratio.",
        "A positive coef (odds_ratio > 1) means the predictor increases the probability that the focal group wins; a negative coef (odds_ratio < 1) means it decreases that probability.",
        "Pay particular attention to the p_value and whether the 95% CI for the coefficient excludes 0 (or for the odds ratio excludes 1) to assess statistical evidence."
    ]
    description = " ".join(description_lines)

    return {"object": result_obj, "description": description}