def extract_final_answer(model_output):
    """
    Extracts statistics for the 'SkinDark' coefficient from a fitted statsmodels results object.
    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coef, SE if available, IRR, IRR CI, p-value, significance boolean)
      - "description": a short plain-language interpretation of those results in the study context.
    """
    import numpy as np

    res = model_output

    # Helper to get item from index-aware or array-like objects
    def get_from_indexable(obj, name, col_index=None):
        # obj can be DataFrame-like with .loc or a ndarray
        try:
            if hasattr(obj, 'loc'):
                # obj.loc[name] may return a Series or array
                row = obj.loc[name]
                if col_index is None:
                    return row
                return row[col_index]
            else:
                # assume numpy array and we need to find the index from params
                if hasattr(res, 'params') and name in list(res.params.index):
                    idx = list(res.params.index).index(name)
                    return obj[idx] if obj.ndim == 1 else obj[idx, col_index]
        except Exception:
            pass
        raise KeyError(f"Could not extract '{name}' from object of type {type(obj)}")

    # Verify presence of parameter
    if not (hasattr(res, 'params') and 'SkinDark' in list(res.params.index)):
        # Try to handle the case where the wrapped results are in an attribute (e.g., .results)
        # but if not found, raise informative error
        raise KeyError("The provided model output does not contain a parameter named 'SkinDark'.")

    # Extract coefficient
    coef = float(get_from_indexable(res.params, 'SkinDark'))

    # Extract p-value if available
    p_value = None
    if hasattr(res, 'pvalues'):
        try:
            p_value = float(get_from_indexable(res.pvalues, 'SkinDark'))
        except Exception:
            p_value = None

    # Extract (cluster-robust) standard error if available
    se = None
    if hasattr(res, 'bse'):
        try:
            se = float(get_from_indexable(res.bse, 'SkinDark'))
        except Exception:
            se = None

    # Extract confidence interval
    ci_lower = ci_upper = None
    if hasattr(res, 'conf_int'):
        try:
            conf = res.conf_int()
            # conf may be DataFrame-like or ndarray
            if hasattr(conf, 'loc'):
                row = conf.loc['SkinDark']
                ci_lower = float(row.iloc[0])
                ci_upper = float(row.iloc[1])
            else:
                idx = list(res.params.index).index('SkinDark')
                ci_lower = float(conf[idx, 0])
                ci_upper = float(conf[idx, 1])
        except Exception:
            ci_lower = ci_upper = None

    # Compute IRR and CI on rate ratio scale
    irr = float(np.exp(coef))
    irr_ci_lower = irr_ci_upper = None
    if (ci_lower is not None) and (ci_upper is not None):
        irr_ci_lower = float(np.exp(ci_lower))
        irr_ci_upper = float(np.exp(ci_upper))

    # Significance at alpha=0.05 if p-value available
    significant = None
    if p_value is not None:
        significant = (p_value < 0.05)

    # Prepare output object
    object_dict = {
        'coef': coef,
        'se': se,
        'p_value': p_value,
        'significant_at_0.05': significant,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper,
    }

    # Build human-readable description
    # Format numbers with sensible precision
    def fmt(x):
        return f"{x:.3f}" if (x is not None) else "NA"

    direction_text = ""
    if significant is True:
        if irr > 1:
            direction_text = "statistically significant evidence that dark-skinned players receive red cards at a higher rate than light-skinned players."
        elif irr < 1:
            direction_text = "statistically significant evidence that dark-skinned players receive red cards at a lower rate than light-skinned players."
        else:
            direction_text = "no difference in the red-card rate between dark- and light-skinned players."
    elif significant is False:
        direction_text = "no statistically significant evidence of a difference in red-card rates between dark- and light-skinned players (at α=0.05)."
    else:
        direction_text = "statistical significance could not be determined (p-value not available)."

    description = (
        "Extracted result for predictor 'SkinDark' from the fitted Poisson model (offset by log(n_matches), "
        "clustered SEs by referee_id). "
        f"Coefficient (log rate ratio) = {fmt(coef)}, SE = {fmt(se)}, p-value = {fmt(p_value)}. "
        f"Incidence rate ratio (IRR) = {fmt(irr)}"
    )
    if (irr_ci_lower is not None) and (irr_ci_upper is not None):
        description += f" with 95% CI = [{fmt(irr_ci_lower)}, {fmt(irr_ci_upper)}]. "
    else:
        description += ". "

    description += f"In plain terms, an IRR of {fmt(irr)} means that, controlling for age, goals, yellow cards, position, and referee-country implicit/explicit bias, "
    description += f"dark-skinned players have {fmt(irr)} times the rate of receiving red cards per match in a player-referee dyad compared to light-skinned players. "
    description += direction_text

    return {"object": object_dict, "description": description}