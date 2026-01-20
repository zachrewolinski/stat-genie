def extract_final_answer(model_output):
    """
    Extract effect of the binary DarkSkin indicator from a fitted statsmodels result object
    (GLMResultsWrapper or robustcov results wrapper).
    
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI for coef, IRR and IRR CI, significance flag)
      - "description": short plain-language interpretation of the effect in the context of the task.
    """
    import numpy as np

    res = model_output

    # Try to access parameter names and statistics in a robust way
    try:
        params = getattr(res, "params")
    except Exception:
        raise ValueError("Provided model_output does not expose .params. Expecting a statsmodels results object.")

    # Find the parameter name corresponding to DarkSkin (try several possible patterns)
    possible_names = ['DarkSkin', 'DarkSkin[T.True]', 'DarkSkin[T.1]', 'DarkSkin_1', 'DarkSkin: 1']
    param_name = None
    for name in possible_names:
        if name in params.index:
            param_name = name
            break
    if param_name is None:
        # fallback: try any parameter containing 'DarkSkin'
        for name in params.index:
            if 'DarkSkin' in str(name):
                param_name = name
                break

    if param_name is None:
        raise KeyError("Could not find a parameter name for DarkSkin in model_output.params. "
                       "Available params: {}".format(list(params.index)))

    # Extract coef, se, pvalue
    coef = float(params[param_name])
    # standard error
    if hasattr(res, 'bse'):
        se = float(getattr(res, 'bse')[param_name])
    else:
        # try .std_errors or similar
        se = float(getattr(res, 'std_errors', getattr(res, 'stderr', np.nan))[param_name])

    # p-value
    if hasattr(res, 'pvalues'):
        pvalue = float(getattr(res, 'pvalues')[param_name])
    else:
        pvalue = float(np.nan)

    # Confidence intervals for the coefficient
    # statsmodels typically provides conf_int() method
    try:
        ci = res.conf_int()
        # conf_int returns a DataFrame/ndarray with rows matching params
        # ensure we can index by param_name
        if hasattr(ci, "loc") and param_name in ci.index:
            ci_low, ci_high = float(ci.loc[param_name, 0]), float(ci.loc[param_name, 1])
        else:
            # assume numpy array with same order as params
            idx = list(params.index).index(param_name)
            ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        # fallback: use coef +/- 1.96*se
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low))
    irr_ci_high = float(np.exp(ci_high))

    # Decide significance at alpha=0.05 if p-value available
    significant_05 = (pvalue < 0.05) if (pvalue is not None and not np.isnan(pvalue)) else None

    # Formulate a short conclusion (yes/no) about whether dark-skinned players are more likely to receive red cards
    if significant_05 is True:
        if irr > 1:
            conclusion = "Yes: dark-skinned players have a statistically significant higher red-card rate per game (IRR > 1, p < 0.05)."
        else:
            conclusion = "No: dark-skinned players have a statistically significant lower red-card rate per game (IRR < 1, p < 0.05)."
    elif significant_05 is False:
        # not statistically significant
        if irr > 1:
            conclusion = "No (not statistically significant): point estimate suggests higher red-card rate for dark-skinned players (IRR > 1) but p >= 0.05."
        elif irr < 1:
            conclusion = "No (not statistically significant): point estimate suggests lower red-card rate for dark-skinned players (IRR < 1) but p >= 0.05."
        else:
            conclusion = "No: point estimate indicates no effect (IRR ~ 1) and result is not statistically significant."
    else:
        conclusion = "Could not determine statistical significance (p-value not available)."

    result_object = {
        "param_name": param_name,
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "95%_CI_coef": [ci_low, ci_high],
        "IRR": irr,
        "95%_CI_IRR": [irr_ci_low, irr_ci_high],
        "significant_at_0.05": significant_05,
        "conclusion_yes_more_likely": True if (significant_05 is True and irr > 1) else False if (significant_05 is not None) else None
    }

    description = (
        f"Effect of DarkSkin ({param_name}) from Negative Binomial model with log(games) offset:\n"
        f"  Coefficient = {coef:.4f} (SE = {se:.4f}), p = {pvalue if (pvalue is not None) else 'NA'}.\n"
        f"  95% CI (coef) = [{ci_low:.4f}, {ci_high:.4f}].\n"
        f"  Exponentiated IRR = {irr:.4f}, 95% CI (IRR) = [{irr_ci_low:.4f}, {irr_ci_high:.4f}].\n\n"
        f"Interpretation: IRR > 1 means dark-skinned players receive red cards at a higher rate per match exposure compared to light-skinned players.\n"
        f"Conclusion: {conclusion}\n"
        f"(Model included offset = log(games) so this compares rates per game; controls and clustering as specified in the model.)"
    )

    return {"object": result_object, "description": description}