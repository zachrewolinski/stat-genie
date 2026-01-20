def extract_final_answer(model_output):
    """
    Extract key inference quantities from a fitted statsmodels OLS RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": dict of extracted numeric results (coefficients, p-values, CIs, estimated
                  effect of livebait when child=0 and when child=1, R^2, n)
      - "description": brief plain-language interpretation of the livebait effect(s)
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Basic model summary numbers
    params = res.params.copy()
    pvalues = res.pvalues.copy()
    bse = res.bse.copy()
    conf_int = res.conf_int(alpha=0.05)
    r_squared = getattr(res, 'rsquared', None)
    nobs = int(getattr(res, 'nobs', getattr(res, 'nobs', None) or getattr(res, 'nobs', 0)))
    df_resid = getattr(res, 'df_resid', None)

    # Helper to robustly find parameter names that correspond to livebait, child, and interaction
    param_names = list(params.index.astype(str))

    def find_param(name_part, exclude_parts=None):
        exclude_parts = exclude_parts or []
        for nm in param_names:
            if name_part in nm and not any(ex in nm for ex in exclude_parts):
                return nm
        return None

    live_name = find_param('livebait', exclude_parts=[':','child' ]) or find_param('livebait', exclude_parts=['child'])
    child_name = find_param('child', exclude_parts=['livebait'])
    interaction_name = None
    # look for any parameter containing both 'livebait' and 'child'
    for nm in param_names:
        if ('livebait' in nm) and ('child' in nm) and (nm != live_name) and (nm != child_name):
            interaction_name = nm
            break

    # If exact names not found, attempt common statsmodels naming patterns:
    if live_name is None:
        # fallback: look for param that contains 'livebait' (maybe encoded like C(livebait)[T.1])
        live_name = find_param('livebait')

    if child_name is None:
        child_name = find_param('child')

    # Build covariance matrix (as DataFrame if possible)
    cov = res.cov_params()
    # Ensure we can index cov by labels; if not, convert to numpy and keep order
    cov_is_df = hasattr(cov, 'loc')

    # Utility to compute linear contrast estimate, se, CI, p-value given a dict of coefficients to sum
    def contrast_stats(coef_terms):
        # coef_terms: dict mapping param_name -> multiplier (usually 1)
        # Build contrast vector in order of params.index
        k = len(param_names)
        c = np.zeros(k, dtype=float)
        for i, nm in enumerate(param_names):
            if nm in coef_terms:
                c[i] = float(coef_terms[nm])
        # estimate
        est = float(np.dot(c, params.values))
        # variance
        if cov_is_df:
            cov_mat = cov.loc[param_names, param_names].values
        else:
            cov_mat = np.asarray(cov)
        var = float(c.dot(cov_mat).dot(c))
        se_ = np.sqrt(var) if var >= 0 else np.nan
        # CI using t critical value if df_resid available, otherwise normal z
        if df_resid is not None:
            t_crit = stats.t.ppf(1 - 0.025, df=df_resid)
            p = float(stats.t.sf(abs(est / se_), df=df_resid) * 2) if se_ > 0 else np.nan
        else:
            t_crit = stats.norm.ppf(1 - 0.025)
            p = float(2 * (1 - stats.norm.cdf(abs(est / se_)))) if se_ > 0 else np.nan
        ci_lower = est - t_crit * se_
        ci_upper = est + t_crit * se_
        return {
            "estimate": est,
            "se": se_,
            "95%_CI": (ci_lower, ci_upper),
            "p_value": p
        }

    # Compute effect of livebait when child = 0 (just the livebait coefficient)
    effect_no_child = None
    if live_name is not None and live_name in params.index:
        effect_no_child = contrast_stats({live_name: 1})
    else:
        # cannot find livebait term
        effect_no_child = {"error": "Could not find a parameter corresponding to 'livebait' in the model."}

    # Compute effect of livebait when child = 1 (livebait + interaction)
    if isinstance(effect_no_child, dict) and "error" in effect_no_child:
        effect_child = {"error": "Cannot compute; livebait parameter missing."}
    else:
        if interaction_name is not None and interaction_name in params.index:
            effect_child = contrast_stats({live_name: 1, interaction_name: 1})
        else:
            # no interaction term present -> same as no-child effect
            effect_child = contrast_stats({live_name: 1})

    # Collect coefficients, p-values, and CIs for all model terms for reference
    coef_table = {}
    for nm in param_names:
        coef_table[nm] = {
            "coef": float(params[nm]),
            "se": float(bse[nm]) if nm in bse.index else None,
            "95%_CI": tuple(conf_int.loc[nm].values) if (hasattr(conf_int, 'loc') and nm in conf_int.index) else None,
            "p_value": float(pvalues[nm]) if nm in pvalues.index else None
        }

    result_object = {
        "n_obs": nobs,
        "df_resid": float(df_resid) if df_resid is not None else None,
        "r_squared": float(r_squared) if r_squared is not None else None,
        "coefficients": coef_table,
        "livebait_effect_child0": effect_no_child,
        "livebait_effect_child1": effect_child,
    }

    # Build a concise description interpreting the key quantities
    if ("error" in effect_no_child) or ("error" in effect_child):
        description = "Could not locate the model parameters for 'livebait' and/or its interaction with 'child'. See 'object' for details."
    else:
        est0 = effect_no_child["estimate"]
        p0 = effect_no_child["p_value"]
        ci0 = effect_no_child["95%_CI"]
        est1 = effect_child["estimate"]
        p1 = effect_child["p_value"]
        ci1 = effect_child["95%_CI"]

        description = (
            f"Estimated effect of using live bait on fish caught per hour when no child is present: "
            f"{est0:.3f} fish/hour (95% CI [{ci0[0]:.3f}, {ci0[1]:.3f}], p = {p0:.3g}). "
            f"When a child is present, the estimated effect of live bait is: "
            f"{est1:.3f} fish/hour (95% CI [{ci1[0]:.3f}, {ci1[1]:.3f}], p = {p1:.3g}). "
            f"Positive estimates indicate that using live bait is associated with catching more fish per hour. "
            f"See 'object[\"coefficients\"]' for coefficients and tests for other covariates (persons, camper, intercept)."
        )

    return {"object": result_object, "description": description}