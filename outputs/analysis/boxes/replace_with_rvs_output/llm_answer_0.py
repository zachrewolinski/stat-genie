def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear and quadratic) and culture-specific age slopes
    from the ClusterRobustResults-like object returned by the model function.
    
    Returns a dictionary:
      {
        "object": {
            "age_linear": {coef, se, z, p, ci_lower, ci_upper},
            "age_quadratic": {coef, se, z, p, ci_lower, ci_upper},
            "per_culture_age_slope": { culture_name: {coef, se, z, p, ci_lower, ci_upper}, ... },
            "interaction_joint_test": { chi2, df, p_value }  # test of whether all age*culture interactions = 0
        },
        "description": "Interpretation text ..."
      }
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Try to obtain parameter names and numeric arrays
    try:
        params = pd.Series(res.params)  # preserves index if it's a Series already
    except Exception:
        # fallback: try to get from original fitted object
        params = pd.Series(getattr(res._orig, 'params', np.array([])))
    param_names = list(params.index)

    # Numeric arrays
    try:
        coef_vals = np.asarray(res.params, dtype=float)
    except Exception:
        coef_vals = np.asarray(params.values, dtype=float)

    try:
        bse = np.asarray(res.bse, dtype=float)
    except Exception:
        # as fallback compute from cov diag if available
        try:
            cov = np.asarray(res.cov)
            bse = np.sqrt(np.diag(cov))
        except Exception:
            bse = np.full_like(coef_vals, np.nan, dtype=float)

    try:
        pvals = np.asarray(res.pvalues, dtype=float)
    except Exception:
        # if unavailable, compute from z-stats
        zstats = coef_vals / bse
        pvals = 2 * (1 - stats.norm.cdf(np.abs(zstats)))

    # conf_int via provided method if available
    try:
        ci = np.asarray(res.conf_int())
        # ensure shape (k,2)
    except Exception:
        crit = stats.norm.ppf(0.975)
        ci = np.column_stack((coef_vals - crit * bse, coef_vals + crit * bse))

    # covariance matrix for linear combinations
    try:
        cov_mat = np.asarray(res.cov)
    except Exception:
        cov_mat = None

    # Helper to get param index
    def idx_of(name):
        try:
            return param_names.index(name)
        except ValueError:
            return None

    # 1) Extract linear age main effect (age_c)
    age_name = 'age_c'
    age_idx = idx_of(age_name)
    age_info = None
    if age_idx is not None:
        age_coef = float(coef_vals[age_idx])
        age_se = float(bse[age_idx]) if age_idx < len(bse) else float('nan')
        age_z = age_coef / age_se if age_se and not np.isnan(age_se) else float('nan')
        age_p = float(pvals[age_idx]) if age_idx < len(pvals) else float('nan')
        age_ci_lower = float(ci[age_idx, 0]) if age_idx < len(ci) else float('nan')
        age_ci_upper = float(ci[age_idx, 1]) if age_idx < len(ci) else float('nan')
        age_info = {
            "coef": age_coef,
            "se": age_se,
            "z": age_z,
            "p": age_p,
            "ci_lower": age_ci_lower,
            "ci_upper": age_ci_upper,
            "interpretation": "Linear (per-unit centered age) effect for the reference culture."
        }

    # 2) Extract quadratic age effect (age_c2)
    age2_name = 'age_c2'
    age2_idx = idx_of(age2_name)
    age2_info = None
    if age2_idx is not None:
        c2_coef = float(coef_vals[age2_idx])
        c2_se = float(bse[age2_idx]) if age2_idx < len(bse) else float('nan')
        c2_z = c2_coef / c2_se if c2_se and not np.isnan(c2_se) else float('nan')
        c2_p = float(pvals[age2_idx]) if age2_idx < len(pvals) else float('nan')
        c2_ci_lower = float(ci[age2_idx, 0]) if age2_idx < len(ci) else float('nan')
        c2_ci_upper = float(ci[age2_idx, 1]) if age2_idx < len(ci) else float('nan')
        age2_info = {
            "coef": c2_coef,
            "se": c2_se,
            "z": c2_z,
            "p": c2_p,
            "ci_lower": c2_ci_lower,
            "ci_upper": c2_ci_upper,
            "interpretation": "Quadratic (age^2) effect: curvature/acceleration of age trend (same across cultures because not interacted)."
        }

    # 3) Identify culture interaction terms with age_c
    # Statsmodels naming for factor interactions typically: 'age_c:C(culture)[T.<level>]'
    interaction_pattern = re.compile(r'^age_c:C\(culture\)\[T\.(.+)\]$')
    interaction_names = []
    interaction_levels = []
    interaction_idxs = []
    for i, name in enumerate(param_names):
        m = interaction_pattern.match(name)
        if m:
            level = m.group(1)
            interaction_names.append(name)
            interaction_levels.append(level)
            interaction_idxs.append(i)

    # Also detect if naming might be 'age_c:C(culture)[Tlevel]' without dot - be robust:
    if not interaction_names:
        alt_pattern = re.compile(r'^age_c:C\(culture\)\[T(.+)\]$')
        for i, name in enumerate(param_names):
            m = alt_pattern.match(name)
            if m:
                level = m.group(1)
                interaction_names.append(name)
                interaction_levels.append(level)
                interaction_idxs.append(i)

    per_culture = dict()
    # Reference culture slope = age_coef
    ref_name = 'reference (omitted)'
    if age_info is not None:
        per_culture[ref_name] = {
            "slope_coef": age_info["coef"],
            "slope_se": float(np.sqrt(cov_mat[age_idx, age_idx])) if cov_mat is not None else age_info["se"],
            "slope_z": age_info["z"],
            "slope_p": age_info["p"],
            "slope_ci_lower": age_info["ci_lower"],
            "slope_ci_upper": age_info["ci_upper"],
            "note": "Reference culture slope (age_c main effect)."
        }

    # For each present interaction, compute culture-specific slope = age_c + interaction
    for name, level, idx in zip(interaction_names, interaction_levels, interaction_idxs):
        inter_coef = float(coef_vals[idx])
        if age_idx is None:
            # cannot compute slope without main age effect
            slope_coef = inter_coef
        else:
            slope_coef = float(coef_vals[age_idx] + inter_coef)

        # Compute variance of sum if covariance matrix available
        if cov_mat is not None and age_idx is not None:
            var_age = cov_mat[age_idx, age_idx]
            var_inter = cov_mat[idx, idx]
            cov_ai = cov_mat[age_idx, idx]
            slope_var = var_age + var_inter + 2.0 * cov_ai
            slope_se = float(np.sqrt(slope_var)) if slope_var >= 0 else float('nan')
        else:
            # fallback approximate: sqrt(se_age^2 + se_inter^2)
            se_age = age_info["se"] if age_info is not None else np.nan
            se_inter = float(bse[idx]) if idx < len(bse) else np.nan
            try:
                slope_se = float(np.sqrt(se_age ** 2 + se_inter ** 2))
            except Exception:
                slope_se = float('nan')

        slope_z = slope_coef / slope_se if slope_se and not np.isnan(slope_se) else float('nan')
        slope_p = 2 * (1 - stats.norm.cdf(abs(slope_z))) if not np.isnan(slope_z) else float('nan')

        # CI
        crit = stats.norm.ppf(0.975)
        slope_ci_lower = slope_coef - crit * slope_se if not np.isnan(slope_se) else float('nan')
        slope_ci_upper = slope_coef + crit * slope_se if not np.isnan(slope_se) else float('nan')

        per_culture[level] = {
            "slope_coef": slope_coef,
            "slope_se": slope_se,
            "slope_z": slope_z,
            "slope_p": slope_p,
            "slope_ci_lower": slope_ci_lower,
            "slope_ci_upper": slope_ci_upper,
            "components": {
                "age_coef": float(coef_vals[age_idx]) if age_idx is not None else None,
                "interaction_coef": inter_coef,
                "interaction_name": name
            }
        }

    # 4) Joint test of all age*culture interactions = 0 (Wald test) if there are any interactions
    interaction_joint = None
    if interaction_idxs:
        try:
            b_vec = coef_vals[interaction_idxs]
            cov_sub = cov_mat[np.ix_(interaction_idxs, interaction_idxs)]
            # handle potential numerical issues with covariance inversion
            inv_cov_sub = np.linalg.pinv(cov_sub)
            w_stat = float(b_vec.T @ inv_cov_sub @ b_vec)
            df_chi = len(interaction_idxs)
            p_chi = 1 - stats.chi2.cdf(w_stat, df_chi)
            interaction_joint = {"chi2": w_stat, "df": df_chi, "p": p_chi,
                                 "interpretation": "Joint test of whether all age*culture interaction coefficients are zero."}
        except Exception:
            interaction_joint = {"chi2": None, "df": len(interaction_idxs), "p": None,
                                 "interpretation": "Could not compute joint test due to numerical issues."}

    # 5) Collect per-interaction individual p-values too
    interaction_individual = {}
    for name, idx in zip(interaction_names, interaction_idxs):
        interaction_individual[name] = {
            "coef": float(coef_vals[idx]),
            "se": float(bse[idx]) if idx < len(bse) else float('nan'),
            "p": float(pvals[idx]) if idx < len(pvals) else float('nan'),
            "ci_lower": float(ci[idx, 0]) if idx < len(ci) else float('nan'),
            "ci_upper": float(ci[idx, 1]) if idx < len(ci) else float('nan')
        }

    # Build the object to return
    obj = {
        "age_linear": age_info,
        "age_quadratic": age2_info,
        "per_culture_age_slope": per_culture,
        "age_by_culture_interaction_individual": interaction_individual,
        "interaction_joint_test": interaction_joint,
        "notes": {
            "reference_culture_label": ref_name,
            "slope_interpretation": "Positive slope => increasing reliance on majority with age; negative => decreasing reliance with age.",
            "quadratic_interpretation": "If age_quadratic coef > 0 the slope accelerates with age; if < 0 slope decelerates (concave)."
        }
    }

    # Short human-readable description
    # We'll summarize main takeaways based on significance heuristics (p < 0.05)
    description_lines = []
    if age_info is not None:
        sig = age_info["p"] < 0.05 if not np.isnan(age_info["p"]) else False
        description_lines.append(
            f"Reference-culture linear age effect: coef={age_info['coef']:.3f}, p={age_info['p']:.3f} "
            f"({'significant' if sig else 'not significant'})."
        )
    if age2_info is not None:
        sig2 = age2_info["p"] < 0.05 if not np.isnan(age2_info["p"]) else False
        description_lines.append(
            f"Quadratic age effect (age^2): coef={age2_info['coef']:.3f}, p={age2_info['p']:.3f} "
            f"({'significant' if sig2 else 'not significant'})."
        )

    if interaction_joint is not None and interaction_joint.get("p") is not None:
        j_sig = interaction_joint["p"] < 0.05
        description_lines.append(
            f"Joint test of age-by-culture interactions: chi2={interaction_joint['chi2']:.3f}, "
            f"df={interaction_joint['df']}, p={interaction_joint['p']:.3f} "
            f"({'evidence that developmental trajectories differ across cultures' if j_sig else 'no strong evidence of differences across cultures'})."
        )
    else:
        description_lines.append("Could not compute a robust joint test of age-by-culture interactions.")

    # Add per-culture summary (concise)
    for cul, info in per_culture.items():
        # skip listing the reference long if many cultures
        description_lines.append(
            f"Culture '{cul}': age slope = {info['slope_coef']:.3f}, se = {info['slope_se']:.3f}, "
            f"p ≈ {info['slope_p']:.3f}."
        )

    description = " ".join(description_lines)

    return {"object": obj, "description": description}