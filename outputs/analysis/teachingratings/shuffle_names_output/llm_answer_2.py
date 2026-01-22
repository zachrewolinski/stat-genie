def extract_final_answer(model_output):
    """
    Extracts estimates and inference for the effect of standardized instructor beauty (prof_z)
    on course evaluation (tenure), including the gender moderation (prof_z:gender_male).
    
    Returns:
      {
        "object": {
           "female_effect": {coef, se, t, p, ci_lower, ci_upper},
           "male_effect":   {coef, se, t, p, ci_lower, ci_upper},
           "interaction":   {coef, se, t, p, ci_lower, ci_upper},
           "notes": "..."
        },
        "description": "Plain-language explanation of the estimated effects and inference."
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Helpful objects
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    cov = res.cov_params()
    ci_df = res.conf_int()  # DataFrame (or ndarray-like) with index matching params
    df_resid = float(res.df_resid) if hasattr(res, "df_resid") else None

    # Identify interaction term name (allow common variants)
    possible_inter_names = [
        "prof_z:gender_male",
        "prof_z*gender_male",
        "gender_male:prof_z",
        "gender_male*prof_z"
    ]
    inter_name = None
    for nm in possible_inter_names:
        if nm in params.index:
            inter_name = nm
            break

    # Require main term
    if "prof_z" not in params.index:
        raise ValueError("Model output does not contain 'prof_z' coefficient.")

    # Extract female (baseline) effect (gender_male = 0)
    coef_f = float(params["prof_z"])
    se_f = float(bse["prof_z"])
    t_f = coef_f / se_f if se_f != 0 else np.nan
    p_f = float(pvals["prof_z"]) if "prof_z" in pvals.index else np.nan
    # CI for main term
    try:
        ci_f_low = float(ci_df.loc["prof_z", 0])
        ci_f_high = float(ci_df.loc["prof_z", 1])
    except Exception:
        # fallback if conf_int returns ndarray
        ci_arr = res.conf_int()
        idx = list(params.index).index("prof_z")
        ci_f_low, ci_f_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])

    female_effect = {
        "coef": coef_f,
        "se": se_f,
        "t": t_f,
        "p": p_f,
        "ci_lower": ci_f_low,
        "ci_upper": ci_f_high,
        "interpretation": (
            "Change in course evaluation score (1-5) associated with a 1 SD increase "
            "in instructor beauty for female instructors (gender_male = 0)."
        ),
    }

    # Extract interaction info (if present)
    if inter_name is not None:
        coef_int = float(params[inter_name])
        se_int = float(bse[inter_name])
        t_int = coef_int / se_int if se_int != 0 else np.nan
        p_int = float(pvals[inter_name]) if inter_name in pvals.index else np.nan
        try:
            ci_int_low = float(ci_df.loc[inter_name, 0])
            ci_int_high = float(ci_df.loc[inter_name, 1])
        except Exception:
            ci_arr = res.conf_int()
            idx = list(params.index).index(inter_name)
            ci_int_low, ci_int_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        interaction = {
            "name": inter_name,
            "coef": coef_int,
            "se": se_int,
            "t": t_int,
            "p": p_int,
            "ci_lower": ci_int_low,
            "ci_upper": ci_int_high,
            "interpretation": (
                "Additional change in the beauty effect for male instructors (the amount "
                "to add to the female baseline effect to get the male effect)."
            ),
        }
    else:
        # If no interaction present, treat interaction as zero
        coef_int = 0.0
        interaction = {
            "name": None,
            "coef": 0.0,
            "se": None,
            "t": None,
            "p": None,
            "ci_lower": None,
            "ci_upper": None,
            "interpretation": "No prof_z x gender_male interaction term found in model.",
        }

    # Compute male effect = coef_f + coef_int, and its SE using covariance matrix
    if inter_name is not None and inter_name in cov.index:
        # var(sum) = var(prof_z) + var(inter) + 2*cov(prof_z, inter)
        var_f = float(cov.loc["prof_z", "prof_z"])
        var_int = float(cov.loc[inter_name, inter_name])
        cov_f_int = float(cov.loc["prof_z", inter_name])
        var_male = var_f + var_int + 2.0 * cov_f_int
        se_male = float(np.sqrt(max(var_male, 0.0)))
        coef_m = coef_f + coef_int
        t_m = coef_m / se_male if se_male != 0 else np.nan
        # p-value using t-distribution with df_resid if available, otherwise normal approx
        if df_resid is not None and df_resid > 0:
            p_m = float(2.0 * stats.t.sf(abs(t_m), df_resid))
            tcrit = stats.t.ppf(1 - 0.025, df_resid)
        else:
            p_m = float(2.0 * stats.norm.sf(abs(t_m)))
            tcrit = stats.norm.ppf(1 - 0.025)
        ci_m_low = coef_m - tcrit * se_male
        ci_m_high = coef_m + tcrit * se_male
    else:
        # If no covariance info for interaction, fall back to sum of coefs and NA for se/p/ci
        coef_m = coef_f + coef_int
        se_male = None
        t_m = None
        p_m = None
        ci_m_low = None
        ci_m_high = None

    male_effect = {
        "coef": coef_m,
        "se": se_male,
        "t": t_m,
        "p": p_m,
        "ci_lower": ci_m_low,
        "ci_upper": ci_m_high,
        "interpretation": (
            "Change in course evaluation score (1-5) associated with a 1 SD increase "
            "in instructor beauty for male instructors (gender_male = 1). "
            "Calculated as main effect + interaction."
        ),
    }

    # Prepare final object
    result_object = {
        "female_effect": female_effect,
        "male_effect": male_effect,
        "interaction": interaction,
        "model_df_resid": df_resid,
        "units": "Outcome (tenure) is course evaluation score (1-5). prof_z is standardized (SD units).",
        "notes": (
            "Estimates and inference use the fitted OLS model provided. The model was "
            "fitted with clustered standard errors by instructor; the SEs, p-values, "
            "and confidence intervals reported for individual coefficients come from that fit. "
            "The male effect SE and p-value above are computed from the covariance matrix "
            "to reflect uncertainty in the linear combination (prof_z + prof_z:gender_male)."
        ),
    }

    # Human-readable description
    # Summarize statistical conclusion about whether beauty affects teaching evals and whether it differs by gender
    desc_lines = []
    desc_lines.append(
        "Interpretation: Coefficients represent change in course evaluation (1-5) per 1 SD increase in instructor beauty."
    )
    # Female
    desc_lines.append(
        f"Female instructors (gender_male=0): estimated effect = {coef_f:.4f} (SE = {se_f:.4f}, "
        f"95% CI [{ci_f_low:.4f}, {ci_f_high:.4f}], p = {p_f:.4f})."
    )
    # Male
    if male_effect["se"] is not None:
        desc_lines.append(
            f"Male instructors (gender_male=1): estimated effect = {coef_m:.4f} (SE = {male_effect['se']:.4f}, "
            f"95% CI [{male_effect['ci_lower']:.4f}, {male_effect['ci_upper']:.4f}], p = {p_m:.4f})."
        )
    else:
        desc_lines.append(
            f"Male instructors (gender_male=1): estimated effect = {coef_m:.4f}. SE/CI/p-value not available due to missing covariance info."
        )
    # Interaction
    if interaction["name"] is not None:
        desc_lines.append(
            f"Interaction (difference between male and female beauty effects): coef = {interaction['coef']:.4f} "
            f"(SE = {interaction['se']:.4f}, 95% CI [{interaction['ci_lower']:.4f}, {interaction['ci_upper']:.4f}], p = {interaction['p']:.4f})."
        )
    else:
        desc_lines.append("No prof_z x gender_male interaction term was found in the model; male and female effects are identical by construction.")
    desc_lines.append("Conclusion: Assess statistical significance from the p-values above (commonly p < 0.05).")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}