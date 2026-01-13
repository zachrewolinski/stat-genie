def extract_final_answer(model_output):
    """
    Extract interpretable statistics about age effects (linear and quadratic) and
    age-by-culture interactions from a fitted statsmodels GLM/Results object.

    Returns a dictionary with:
      - "object": dict with coefficients, standard errors, p-values, 95% CIs for:
          * age_c (linear)
          * age_sq (quadratic)
          * per-culture linear slopes for age (combined age_c + interaction when present),
            with inference computed by linear combination using the covariance matrix.
        Also includes which culture was used as the reference (if that can be inferred).
      - "description": a short plain-language interpretation of the extracted results
        in the context of the task (how reliance on majority preference develops with age,
        and whether trajectories differ across cultures).
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Basic parameter table
    try:
        params = res.params.copy()
    except Exception:
        raise ValueError("model_output does not expose .params")

    # Try to get covariance matrix for linear combinations
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Helper to safely get bse, pvalues, conf_int
    bse = getattr(res, "bse", None)
    pvalues = getattr(res, "pvalues", None)
    try:
        ci_table = res.conf_int()
    except Exception:
        ci_table = None

    param_names = list(params.index.astype(str))

    # Find primary terms
    def _get_param(name):
        if name in params.index:
            coef = float(params[name])
            se = float(bse[name]) if (bse is not None and name in bse.index) else np.nan
            p = float(pvalues[name]) if (pvalues is not None and name in pvalues.index) else np.nan
            if ci_table is not None and name in ci_table.index:
                ci_low, ci_high = float(ci_table.loc[name, 0]), float(ci_table.loc[name, 1])
            else:
                ci_low, ci_high = (coef - 1.96 * se, coef + 1.96 * se) if not np.isnan(se) else (np.nan, np.nan)
            return {"coef": coef, "se": se, "p": p, "ci95": (ci_low, ci_high)}
        else:
            return None

    age_linear = _get_param("age_c")
    age_quad = _get_param("age_sq")

    # Identify culture-related parameter names for main effects and interactions
    # Patterns in statsmodels: "C(culture)[T.level]" and "age_c:C(culture)[T.level]"
    culture_main_params = [n for n in param_names if n.startswith("C(culture)")]
    age_c_inter_params = [n for n in param_names if ("age_c" in n) and ("C(culture)" in n)]

    # Attempt to infer culture levels and reference
    culture_levels = []
    for n in culture_main_params:
        # extract level between [T.  and ]
        try:
            start = n.index("[T.") + 3
            end = n.index("]", start)
            lvl = n[start:end]
        except Exception:
            # fallback: take after last '[' until ']'
            try:
                start = n.rindex("[") + 1
                end = n.rindex("]")
                lvl = n[start:end]
            except Exception:
                lvl = n
        culture_levels.append(lvl)

    # For interactions, extract the level similarly
    inter_levels = []
    for n in age_c_inter_params:
        try:
            start = n.index("[T.") + 3
            end = n.index("]", start)
            lvl = n[start:end]
        except Exception:
            try:
                start = n.rindex("[") + 1
                end = n.rindex("]")
                lvl = n[start:end]
            except Exception:
                lvl = n
        inter_levels.append((lvl, n))

    # Try to get full list of culture categories from the original data if available,
    # so we can identify the reference (the one NOT appearing as a param with [T.level]).
    reference_culture = None
    try:
        df = res.model.data.frame
        if "culture" in df.columns:
            unique_cult = list(map(str, pd.unique(df["culture"])))
            # The design matrix encodes all levels except reference with C(culture)[T.level];
            # find which unique_cult is not among the 'culture_levels' extracted above.
            missing = [c for c in unique_cult if c not in culture_levels]
            if len(missing) == 1:
                reference_culture = missing[0]
            elif len(missing) > 1:
                # Heuristic: statsmodels uses the first category (sorted) as reference unless specified;
                # choose the first of unique_cult as reference if multiple missing.
                reference_culture = missing[0]
            else:
                reference_culture = None
    except Exception:
        # If we couldn't access the dataframe, leave reference_culture as None
        reference_culture = None

    # If we still don't know the reference culture, try to infer from parameter names:
    if reference_culture is None:
        # If we found any interaction levels, then reference is "the level not listed" -
        # we cannot know its name. Indicate unknown.
        reference_culture = "(reference level not present in params or unavailable from data)"

    # Compute per-culture linear slopes for age:
    # - For reference culture: slope = age_c
    # - For each interaction level L: slope_L = age_c + coef(age_c:C(culture)[T.L])
    culture_slopes = {}
    if age_linear is None:
        raise ValueError("Model does not contain an 'age_c' parameter; cannot compute age slopes.")
    base_coef = age_linear["coef"]
    # Variance of base
    if cov is not None and "age_c" in cov.index:
        var_base = float(cov.loc["age_c", "age_c"])
    else:
        var_base = age_linear["se"] ** 2 if not np.isnan(age_linear["se"]) else np.nan

    # add reference culture entry
    culture_slopes[reference_culture] = {
        "slope_coef": base_coef,
        "slope_se": float(np.sqrt(var_base)) if not np.isnan(var_base) else np.nan,
        "slope_p": float(age_linear["p"]) if age_linear is not None else np.nan,
        "slope_ci95": (float(base_coef - 1.96 * np.sqrt(var_base)), float(base_coef + 1.96 * np.sqrt(var_base))) if not np.isnan(var_base) else (np.nan, np.nan),
        "note": "Reference culture (no interaction term in params)."
    }

    # For each interaction level compute combined slope and inference via covariance
    for lvl, pname in inter_levels:
        inter = _get_param(pname)
        if inter is None:
            continue
        inter_coef = inter["coef"]
        # slope = base_coef + inter_coef
        slope = base_coef + inter_coef
        # compute variance: Var(age_c) + Var(inter) + 2*Cov(age_c, inter)
        if cov is not None and ("age_c" in cov.index) and (pname in cov.index):
            var_inter = float(cov.loc[pname, pname])
            covar = float(cov.loc["age_c", pname])
            var_slope = var_base + var_inter + 2.0 * covar
        else:
            # fallback: approximate by summing variances (conservative, ignores covariance)
            var_inter = inter["se"] ** 2 if not np.isnan(inter["se"]) else np.nan
            if np.isnan(var_inter) or np.isnan(var_base):
                var_slope = np.nan
            else:
                var_slope = var_base + var_inter
        se_slope = float(np.sqrt(var_slope)) if not np.isnan(var_slope) else np.nan
        # compute z and p-value if possible
        if not np.isnan(se_slope) and se_slope > 0:
            z = slope / se_slope
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            ci_low = slope - 1.96 * se_slope
            ci_high = slope + 1.96 * se_slope
        else:
            p = np.nan
            ci_low, ci_high = (np.nan, np.nan)
        culture_slopes[lvl] = {
            "slope_coef": float(slope),
            "slope_se": se_slope,
            "slope_p": float(p) if not np.isnan(p) else np.nan,
            "slope_ci95": (float(ci_low) if not np.isnan(ci_low) else np.nan,
                           float(ci_high) if not np.isnan(ci_high) else np.nan),
            "interaction_param": pname,
            "interaction_coef": inter_coef,
            "interaction_se": inter["se"],
            "interaction_p": inter["p"]
        }

    # Summarize quadratic term interpretation
    quad_summary = None
    if age_quad is not None:
        quad_summary = {
            "coef": age_quad["coef"],
            "se": age_quad["se"],
            "p": age_quad["p"],
            "ci95": age_quad["ci95"],
        }

    # Build object to return
    result_object = {
        "age_linear": age_linear,
        "age_quadratic": quad_summary,
        "culture_reference": reference_culture,
        "culture_slopes_for_age": culture_slopes,
        "notes": (
            "Slopes for each culture represent the linear effect of centered age on log-odds of "
            "choosing the majority option. A positive slope means increasing reliance on the majority with age. "
            "Quadratic term (age_sq) is global across cultures in this model and indicates acceleration/deceleration."
        )
    }

    # Short plain-language description
    # We'll highlight whether age has a significant global linear effect and whether any culture slopes differ
    desc_lines = []
    try:
        age_p = age_linear["p"]
        age_coef = age_linear["coef"]
        if not np.isnan(age_p) and age_p < 0.05:
            desc_lines.append(f"Overall (reference culture) linear age effect: coef={age_coef:.3f} (p={age_p:.3f}) — significant. This indicates that reliance on the majority changes with age in the reference culture.")
        else:
            desc_lines.append(f"Overall (reference culture) linear age effect: coef={age_coef:.3f} (p={age_p:.3f}) — not statistically significant.")
    except Exception:
        desc_lines.append("Could not determine significance for the overall linear age effect.")

    if quad_summary is not None:
        q_p = quad_summary["p"]
        q_coef = quad_summary["coef"]
        if not np.isnan(q_p) and q_p < 0.05:
            desc_lines.append(f"Quadratic age effect (age_sq): coef={q_coef:.3f} (p={q_p:.3f}) — significant, indicating a non-linear developmental trajectory across ages.")
        else:
            desc_lines.append(f"Quadratic age effect (age_sq): coef={q_coef:.3f} (p={q_p:.3f}) — not significant.")

    # Check interactions: list cultures whose slopes differ significantly from reference
    differing = []
    for lvl, info in culture_slopes.items():
        # skip reference entry
        if lvl == reference_culture:
            continue
        pval = info.get("slope_p", np.nan)
        # But slope_p tests whether slope differs from zero for that culture; better to test interaction param p-value:
        inter_p = info.get("interaction_p", np.nan)
        if not np.isnan(inter_p) and inter_p < 0.05:
            differing.append((lvl, float(inter_p)))
    if len(differing) > 0:
        desc_lines.append("There is evidence that developmental slopes differ across cultures. Significant interactions (p<0.05) were found for: " +
                          ", ".join([f"{lvl} (interaction p={p:.3f})" for lvl, p in differing]) + ".")
    else:
        desc_lines.append("No culture showed a significant interaction with age at p<0.05 based on the interaction parameters; developmental slopes do not strongly differ across cultures in this model.")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}