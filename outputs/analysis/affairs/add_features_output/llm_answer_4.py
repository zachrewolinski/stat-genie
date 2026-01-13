def extract_final_answer(model_output):
    """
    Extracts the effect of 'has_children' on the count of extramarital affairs from a
    fitted statsmodels ZeroInflatedNegativeBinomialResultsWrapper.

    Returns a dictionary with:
      - "object": dict with numeric summaries for:
          * female effect of has_children (count model)
          * male effect of has_children (count model, using interaction children_x_male)
          * inflation (logit) effect of has_children (if present)
      - "description": short text interpretation of what the numbers mean.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    params = res.params
    bse = res.bse
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Helper to find parameter names robustly
    def find_param(name):
        # exact match first
        if name in params.index:
            return name
        # try variants
        for idx in params.index:
            if idx.endswith(name):
                return idx
            if name in idx:
                return idx
        return None

    # Count-model parameter names
    has_children_name = find_param('has_children')
    interaction_name = find_param('children_x_male') or find_param('has_children:gender_male') or find_param('has_children:gender')  # try common variants

    # Inflation-model parameter name (often prefixed 'inflate_')
    inflate_has_children_name = None
    for idx in params.index:
        if 'inflate' in idx and 'has_children' in idx:
            inflate_has_children_name = idx
            break
    # fallback: if there's a second block of params, try to detect inflation param by suffix/pattern
    if inflate_has_children_name is None:
        # sometimes inflation params are after count params and may be named like 'has_children' but with a suffix/prefix
        for idx in params.index:
            if idx != has_children_name and 'has_children' in idx:
                inflate_has_children_name = idx
                break

    output = {}

    # ----- Count model: female effect (gender_male = 0) -----
    if has_children_name is None:
        raise ValueError("Could not find a parameter named 'has_children' in model params.")
    coef_f = float(params[has_children_name])
    se_f = float(bse[has_children_name]) if has_children_name in bse.index else np.nan
    z_f = coef_f / se_f if se_f and not np.isnan(se_f) else np.nan
    p_f = 2.0 * (1.0 - stats.norm.cdf(abs(z_f))) if not np.isnan(z_f) else np.nan
    irr_f = np.exp(coef_f)
    ci_lower_f = np.exp(coef_f - 1.96 * se_f) if not np.isnan(se_f) else np.nan
    ci_upper_f = np.exp(coef_f + 1.96 * se_f) if not np.isnan(se_f) else np.nan

    output['female_effect_count'] = {
        'param_name': has_children_name,
        'coef': coef_f,
        'se': se_f,
        'z': z_f,
        'p_value': p_f,
        'IRR': irr_f,
        'IRR_95CI': (ci_lower_f, ci_upper_f),
        'interpretation': (
            "This is the log count (log incidence rate) effect of having children for females "
            "(gender_male=0). IRR < 1 implies fewer expected affairs when having children."
        )
    }

    # ----- Count model: male effect (gender_male = 1) -----
    if interaction_name is None:
        # No interaction found: male effect is same as female effect plus (if gender main effect)
        # But since user specified an interaction, we try to handle gracefully.
        coef_m = coef_f
        # use same se (conservative)
        se_m = se_f
        z_m = z_f
        p_m = p_f
        irr_m = irr_f
        ci_lower_m, ci_upper_m = ci_lower_f, ci_upper_f
        note = "No children_x_male interaction found; male effect assumed equal to female effect."
    else:
        coef_int = float(params[interaction_name])
        # combined coefficient = coef_f + coef_int
        coef_m = coef_f + coef_int
        # compute SE for sum using covariance if available
        if cov is not None and has_children_name in cov.index and interaction_name in cov.index:
            var_sum = cov.loc[has_children_name, has_children_name] + cov.loc[interaction_name, interaction_name] + 2.0 * cov.loc[has_children_name, interaction_name]
            se_m = float(np.sqrt(max(var_sum, 0.0)))
        else:
            # fallback: approximate by sqrt(se_f^2 + se_int^2)
            se_int = float(bse[interaction_name]) if interaction_name in bse.index else np.nan
            se_m = float(np.sqrt(max((se_f or 0.0)**2 + (se_int or 0.0)**2, 0.0)))
        z_m = coef_m / se_m if se_m and not np.isnan(se_m) else np.nan
        p_m = 2.0 * (1.0 - stats.norm.cdf(abs(z_m))) if not np.isnan(z_m) else np.nan
        irr_m = np.exp(coef_m)
        ci_lower_m = np.exp(coef_m - 1.96 * se_m) if not np.isnan(se_m) else np.nan
        ci_upper_m = np.exp(coef_m + 1.96 * se_m) if not np.isnan(se_m) else np.nan
        note = "Male effect computed as sum of has_children (count) and the children_x_male interaction."

    output['male_effect_count'] = {
        'param_names': (has_children_name, interaction_name),
        'coef': coef_m,
        'se': se_m,
        'z': z_m,
        'p_value': p_m,
        'IRR': irr_m,
        'IRR_95CI': (ci_lower_m, ci_upper_m),
        'note': note,
        'interpretation': (
            "This is the log count effect of having children for males (gender_male=1). "
            "IRR < 1 implies fewer expected affairs when having children."
        )
    }

    # ----- Inflation (logit) effect of has_children -----
    if inflate_has_children_name is not None and inflate_has_children_name in params.index:
        coef_infl = float(params[inflate_has_children_name])
        se_infl = float(bse[inflate_has_children_name]) if inflate_has_children_name in bse.index else np.nan
        z_infl = coef_infl / se_infl if se_infl and not np.isnan(se_infl) else np.nan
        p_infl = 2.0 * (1.0 - stats.norm.cdf(abs(z_infl))) if not np.isnan(z_infl) else np.nan
        or_infl = np.exp(coef_infl)
        ci_lower_infl = np.exp(coef_infl - 1.96 * se_infl) if not np.isnan(se_infl) else np.nan
        ci_upper_infl = np.exp(coef_infl + 1.96 * se_infl) if not np.isnan(se_infl) else np.nan

        output['inflation_has_children'] = {
            'param_name': inflate_has_children_name,
            'coef': coef_infl,
            'se': se_infl,
            'z': z_infl,
            'p_value': p_infl,
            'OR': or_infl,
            'OR_95CI': (ci_lower_infl, ci_upper_infl),
            'interpretation': (
                "This is the log-odds effect of having children on being an 'excess' zero (structural non-participant). "
                "OR < 1 implies having children is associated with lower odds of being in the always-zero group (i.e., "
                "less likely to be a structural non-participant)."
            )
        }
    else:
        output['inflation_has_children'] = None

    # Short textual description summarizing the results for the question:
    # Determine whether having children decreases engagement in affairs based on IRR and p-values.
    def interpret_decision(entry):
        if entry is None:
            return "No inflation term for has_children detected."
        coef = entry['coef']
        p = entry['p_value']
        irr = entry['OR']
        # For inflation we don't use this function; remain generic
        return None

    desc_lines = []
    # Female
    ff = output['female_effect_count']
    desc_lines.append(
        f"For females (gender_male=0): has_children coef={ff['coef']:.4f}, SE={ff['se']:.4f}, p={ff['p_value']:.3g}. "
        f"IRR={ff['IRR']:.3f} (95% CI {ff['IRR_95CI'][0]:.3f} to {ff['IRR_95CI'][1]:.3f})."
    )
    # Male
    mm = output['male_effect_count']
    desc_lines.append(
        f"For males (gender_male=1): has_children combined coef={mm['coef']:.4f}, SE={mm['se']:.4f}, p={mm['p_value']:.3g}. "
        f"IRR={mm['IRR']:.3f} (95% CI {mm['IRR_95CI'][0]:.3f} to {mm['IRR_95CI'][1]:.3f})."
    )
    # Inflation
    infl = output['inflation_has_children']
    if infl is not None:
        desc_lines.append(
            f"Inflation-part (logit) for has_children: coef={infl['coef']:.4f}, SE={infl['se']:.4f}, p={infl['p_value']:.3g}. "
            f"OR={infl['OR']:.3f} (95% CI {infl['OR_95CI'][0]:.3f} to {infl['OR_95CI'][1]:.3f})."
        )
    else:
        desc_lines.append("No inflation (logit) parameter for has_children was found in the model output.")

    # Overall interpretive summary (conservative): look at significance and direction for male & female
    summary_interp = []
    for label, part in [('females', ff), ('males', mm)]:
        sign = 'decrease' if part['IRR'] < 1 else ('increase' if part['IRR'] > 1 else 'no change')
        sig = 'statistically significant' if (part['p_value'] is not None and part['p_value'] < 0.05) else 'not statistically significant'
        summary_interp.append(f"For {label}: having children is associated with a {sign} in the expected count of affairs ({sig}; p={part['p_value']:.3g}).")

    desc_lines.extend(summary_interp)

    description = " ".join(desc_lines)

    return {"object": output, "description": description}