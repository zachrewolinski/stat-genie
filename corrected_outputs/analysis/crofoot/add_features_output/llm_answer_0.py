def extract_final_answer(model_output):
    """
    Extract statistics relevant to how relative group size and contest location
    (and their interaction) influence the probability of the focal group winning.

    Returns a dictionary with:
      - "object": a dict of extracted numeric results (coefficients, SE, z, p, 95% CI,
                  odds-ratios, and marginal effects of size at dist = -1, 0, +1 SD)
      - "description": a brief interpretation of those results in the study context.

    Expects a statsmodels GLMResults-like object (e.g., GLMResultsWrapper).
    """
    import numpy as np
    from math import exp
    try:
        from scipy import stats as _stats
        norm_cdf = _stats.norm.cdf
    except Exception:
        # fallback: use normal cdf via numpy (approximation using erfc)
        def norm_cdf(x):
            return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

    # Helper to safely get parameter names as used by statsmodels formula API
    name_size = "size_diff_z"
    name_dist = "dist_diff_z"
    name_inter = f"{name_size}:{name_dist}"  # expected interaction name in statsmodels

    params = model_output.params
    pvalues = getattr(model_output, "pvalues", None)
    bse = None
    try:
        # GLMResults has bse attribute (std errors)
        bse = model_output.bse
    except Exception:
        # fallback to cov_params
        cov = model_output.cov_params()
        bse = np.sqrt(np.diag(cov))

    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        # try as attribute or method
        try:
            cov = model_output.cov_params
        except Exception:
            cov = None

    # Safely extract coefficients; if interaction not present, treat as zero and warn in description
    def get_param(name):
        if name in params.index:
            coef = float(params.loc[name])
            se = float(bse.loc[name]) if hasattr(bse, "loc") else float(bse[params.index.get_loc(name)])
            p = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
            # 95% CI from model if available:
            try:
                ci_low, ci_high = model_output.conf_int().loc[name]
                ci = (float(ci_low), float(ci_high))
            except Exception:
                ci = (coef - 1.96 * se, coef + 1.96 * se)
            return {"coef": coef, "se": se, "p": p, "ci95": ci}
        else:
            return None

    info_size = get_param(name_size)
    info_dist = get_param(name_dist)
    info_inter = get_param(name_inter)

    # If interaction uses reverse ordering (dist_diff_z:size_diff_z), try that name
    if info_inter is None:
        alt_inter = f"{name_dist}:{name_size}"
        info_inter = get_param(alt_inter)
        if info_inter is not None:
            name_inter = alt_inter

    # If still None, set zeros
    missing_interaction = False
    if info_size is None:
        raise KeyError(f"Model does not contain expected predictor '{name_size}'. Available params: {list(params.index)}")
    if info_dist is None:
        raise KeyError(f"Model does not contain expected predictor '{name_dist}'. Available params: {list(params.index)}")
    if info_inter is None:
        # treat interaction as zero (no interaction estimated)
        missing_interaction = True
        # create a zeroed structure with large SE = NaN to indicate absence
        info_inter = {"coef": 0.0, "se": float("nan"), "p": None, "ci95": (float("nan"), float("nan"))}

    # Prepare summary entries for main coefficients
    def make_entry(name, info):
        return {
            "term": name,
            "coef": info["coef"],
            "se": info["se"],
            "p": info["p"],
            "ci95": info["ci95"],
            "odds_ratio": exp(info["coef"]),
            "or_ci95": (exp(info["ci95"][0]), exp(info["ci95"][1]))
        }

    entry_size = make_entry(name_size, info_size)
    entry_dist = make_entry(name_dist, info_dist)
    entry_inter = make_entry(name_inter, info_inter)

    # Compute marginal (conditional) effect of size at representative values of dist_diff_z: -1, 0, +1 (SD units)
    # effect (log-odds) = beta_size + beta_inter * dist_val
    # SE via delta method: Var = Var(beta_size) + dist_val^2 Var(beta_inter) + 2*dist_val*Cov(beta_size,beta_inter)
    marg_effects = {}
    # Acquire covariances if available
    cov_matrix = None
    if cov is not None:
        # If cov is a DataFrame-like
        try:
            cov_matrix = cov
        except Exception:
            cov_matrix = None

    # function to compute combined effect and SE
    def combined_effect(dist_val):
        b_size = info_size["coef"]
        b_inter = info_inter["coef"]
        eff = b_size + b_inter * dist_val
        # compute SE
        se_eff = None
        if cov_matrix is not None and name_size in cov_matrix.index and name_inter in cov_matrix.index:
            var_size = float(cov_matrix.loc[name_size, name_size])
            var_inter = float(cov_matrix.loc[name_inter, name_inter])
            covar = float(cov_matrix.loc[name_size, name_inter])
            var_eff = var_size + (dist_val ** 2) * var_inter + 2.0 * dist_val * covar
            if var_eff < 0:
                se_eff = float("nan")
            else:
                se_eff = float(np.sqrt(var_eff))
        else:
            # fallback: propagate using available SEs but assume cov=0 (conservative/approx)
            se_size = info_size["se"]
            se_inter = info_inter["se"]
            if np.isfinite(se_size) and np.isfinite(se_inter):
                var_eff = se_size ** 2 + (dist_val ** 2) * (se_inter ** 2)
                se_eff = float(np.sqrt(var_eff))
            else:
                se_eff = float("nan")

        # z & p
        if se_eff and not np.isnan(se_eff) and se_eff > 0:
            z = eff / se_eff
            p = 2.0 * (1.0 - norm_cdf(abs(z)))
            ci_low = eff - 1.96 * se_eff
            ci_high = eff + 1.96 * se_eff
        else:
            z = float("nan")
            p = None
            ci_low, ci_high = (float("nan"), float("nan"))

        return {
            "dist_value": dist_val,
            "log_odds_change_per_1SD_size": eff,
            "se": se_eff,
            "z": z,
            "p": p,
            "ci95_log_odds": (ci_low, ci_high),
            "odds_ratio_per_1SD_size": exp(eff) if np.isfinite(eff) else float("nan"),
            "or_ci95": (exp(ci_low) if np.isfinite(ci_low) else float("nan"),
                        exp(ci_high) if np.isfinite(ci_high) else float("nan"))
        }

    for dv in [-1.0, 0.0, 1.0]:
        marg_effects[str(dv)] = combined_effect(dv)

    # Summarize interaction significance
    interaction_significant = (info_inter["p"] is not None) and (info_inter["p"] < 0.05)

    result_object = {
        "model_params_available": list(params.index),
        "size_term": entry_size,
        "dist_term": entry_dist,
        "size_by_dist_term": entry_inter,
        "interaction_estimated": not missing_interaction,
        "interaction_significant_p_lt_0_05": bool(interaction_significant),
        "marginal_effects_size_at_dist_values": marg_effects,
        "notes": (
            "Marginal effects computed for a 1 SD increase in relative focal group size "
            "at contest location advantages (dist_diff_z) of -1, 0, +1 SD. "
            "Odds ratios are exp(log-odds). If dyad fixed effects are present, "
            "intercepts vary by dyad; the reported marginal 'log-odds change' is "
            "the change in log-odds of focal group winning associated with a 1 SD increase "
            "in size_diff_z at the specified dist_diff_z value, holding other numeric covariates at 0."
        )
    }

    description_lines = [
        "Extracted key coefficients and tests for the predictors of interest (size_diff_z, dist_diff_z, and their interaction).",
        f"size_diff_z coefficient = {entry_size['coef']:.4g} (SE={entry_size['se']:.4g}), p = {entry_size['p']}",
        f"dist_diff_z coefficient = {entry_dist['coef']:.4g} (SE={entry_dist['se']:.4g}), p = {entry_dist['p']}",
    ]
    if missing_interaction:
        description_lines.append(
            "No interaction term was estimated in the model (size:dist not present); marginal effects assume no interaction."
        )
    else:
        description_lines.append(
            f"Interaction ({name_inter}) coefficient = {entry_inter['coef']:.4g} (SE={entry_inter['se']:.4g}), p = {entry_inter['p']}"
        )
        if interaction_significant:
            description_lines.append("The size-by-location interaction is statistically significant (p < 0.05), indicating that the effect of relative group size on winning depends on contest location.")
        else:
            description_lines.append("The size-by-location interaction is not statistically significant (p >= 0.05), so there is no strong evidence that the size effect differs by location.")

    description_lines.append(
        "Marginal effects: for each of dist_diff_z = -1, 0, +1 (SD), the function reports the change in log-odds of winning per 1 SD increase in size, its SE, p, 95% CI, and corresponding odds ratios."
    )

    description = " ".join(description_lines)

    return {"object": result_object, "description": description}