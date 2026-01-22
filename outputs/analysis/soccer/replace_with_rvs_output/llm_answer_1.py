def extract_final_answer(model_output):
    """
    Extract statistics for the 'SkinDark' coefficient from the clustered model output.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, z, p, 95% CI, IRR and IRR CI, significance flag, n_clusters)
      - "description": plain-language interpretation about whether dark-skinned players are more likely
                       than light-skinned players to receive red cards (based on sign and significance).

    The function is written to be robust to variations in how the model_output exposes parameters
    (pandas Series or numpy arrays) and to different parameter name encodings (e.g., 'SkinDark',
    'SkinDark[T.True]', etc.).
    """
    import numpy as np
    import math
    try:
        from scipy import stats as sps
    except Exception:
        sps = None

    # Helper to safely get attribute or fallback to underlying results
    def safe_get(attr_name):
        if hasattr(model_output, attr_name):
            return getattr(model_output, attr_name)
        # fallback to underlying results if available
        if hasattr(model_output, "_results") and hasattr(model_output._results, attr_name):
            return getattr(model_output._results, attr_name)
        return None

    params = safe_get("params")
    if params is None:
        raise ValueError("Could not find parameters (params) on model_output.")

    # Identify the parameter name corresponding to skin tone
    # Try several likely names or search for substring 'SkinDark'
    param_names = list(params.index) if hasattr(params, "index") else None
    skin_param_name = None
    if param_names:
        candidates = [n for n in param_names if "SkinDark" in str(n)]
        if len(candidates) >= 1:
            # prefer exact match if present
            if "SkinDark" in param_names:
                skin_param_name = "SkinDark"
            else:
                skin_param_name = candidates[0]
    else:
        # params may be a numpy array (unlikely); then we cannot identify by name
        raise ValueError("Parameter names unavailable on model_output.params; cannot locate SkinDark parameter.")

    if skin_param_name is None:
        raise ValueError("Could not locate a parameter name containing 'SkinDark' in model_output.params index.")

    # Extract coefficient
    coef = float(params[skin_param_name])

    # Extract clustered standard error (model_output.bse may be series or array)
    bse_all = safe_get("bse")
    if bse_all is None:
        raise ValueError("Could not find clustered standard errors (bse) on model_output.")
    # If bse_all has index, use same name; else find position from params.index
    if hasattr(bse_all, "get") or (hasattr(bse_all, "index") if bse_all is not None else False):
        # pandas Series-like
        try:
            se = float(bse_all[skin_param_name])
        except Exception:
            # fallback: try alignment by position
            loc = list(params.index).index(skin_param_name)
            se = float(np.asarray(bse_all)[loc])
    else:
        # numpy array-like: align by position
        loc = list(params.index).index(skin_param_name)
        se = float(np.asarray(bse_all)[loc])

    # z and p: try to get precomputed ones, else compute
    z_all = safe_get("zvalues")
    p_all = safe_get("pvalues")
    if z_all is not None:
        try:
            z = float(z_all[skin_param_name]) if hasattr(z_all, "index") else float(np.asarray(z_all)[list(params.index).index(skin_param_name)])
        except Exception:
            z = coef / se
    else:
        z = coef / se

    if p_all is not None:
        try:
            p = float(p_all[skin_param_name]) if hasattr(p_all, "index") else float(np.asarray(p_all)[list(params.index).index(skin_param_name)])
        except Exception:
            # compute from z
            if sps is not None:
                p = 2 * (1 - sps.norm.cdf(abs(z)))
            else:
                p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        if sps is not None:
            p = 2 * (1 - sps.norm.cdf(abs(z)))
        else:
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    # Confidence interval: try to extract model_output.conf_int (clustered)
    conf_all = safe_get("conf_int")
    if conf_all is not None:
        # conf_all might be a 2D array aligned with params order
        try:
            if hasattr(conf_all, "loc") and skin_param_name in conf_all.index:
                ci_row = conf_all.loc[skin_param_name].values
                ci_low, ci_high = float(ci_row[0]), float(ci_row[1])
            else:
                # assume array with same ordering as params.index
                arr = np.asarray(conf_all)
                loc = list(params.index).index(skin_param_name)
                ci_low, ci_high = float(arr[loc, 0]), float(arr[loc, 1])
        except Exception:
            # fallback to coef +/- 1.96*se
            z_crit = 1.96
            ci_low, ci_high = coef - z_crit * se, coef + z_crit * se
    else:
        z_crit = 1.96
        ci_low, ci_high = coef - z_crit * se, coef + z_crit * se

    # Incidence rate ratio (IRR) and CI on IRR scale
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low))
    irr_ci_high = float(np.exp(ci_high))

    # Count clusters if available
    cluster_groups = safe_get("cluster_groups")
    n_clusters = None
    if cluster_groups is not None:
        try:
            n_clusters = int(np.unique(np.asarray(cluster_groups)).size)
        except Exception:
            n_clusters = None

    # Statistical significance (two-sided alpha=0.05)
    significant = (p < 0.05)

    # Build the object to return
    result_obj = {
        "parameter_name": skin_param_name,
        "coef_log_rate": coef,
        "std_error_clustered": se,
        "z_value": z,
        "p_value": p,
        "ci_95_log_rate": [ci_low, ci_high],
        "irr": irr,
        "irr_95_ci": [irr_ci_low, irr_ci_high],
        "significant_at_0.05": bool(significant),
        "n_clusters_referees": n_clusters,
    }

    # Prepare a concise interpretation
    direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
    if significant:
        if coef > 0:
            conclusion = ("There is a statistically significant positive association: dark-skinned players "
                          "are estimated to receive more red cards per game than light-skinned players.")
        elif coef < 0:
            conclusion = ("There is a statistically significant negative association: dark-skinned players "
                          "are estimated to receive fewer red cards per game than light-skinned players.")
        else:
            conclusion = "Estimated effect is exactly zero."
    else:
        conclusion = ("The estimated association is not statistically significant at alpha=0.05; "
                      "the data do not provide strong evidence that dark-skinned players receive a different "
                      "rate of red cards per game than light-skinned players.")

    description = (
        f"Parameter '{skin_param_name}': log-rate coefficient = {coef:.4f} (clustered SE = {se:.4f}), "
        f"z = {z:.3f}, p = {p:.3g}, 95% CI for log-rate = [{ci_low:.4f}, {ci_high:.4f}]. "
        f"On the incidence-rate (multiplicative) scale: IRR = {irr:.3f}, 95% CI = [{irr_ci_low:.3f}, {irr_ci_high:.3f}]. "
        f"Interpretation: the coefficient is the log of the rate ratio (offset = log(games)), so IRR>1 means dark-skinned "
        f"players receive more red cards per game. {conclusion} "
        f"Number of referee clusters used for clustering SEs: {n_clusters}."
    )

    return {"object": result_obj, "description": description}