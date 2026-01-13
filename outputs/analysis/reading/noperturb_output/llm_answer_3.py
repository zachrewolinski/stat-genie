def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals and
    marginal effects for the Reader View treatment and its interaction with dyslexia.
    
    Returns:
      {
        "object": <dict of numeric results>,
        "description": <text interpretation>
      }
    """
    import numpy as np
    import pandas as pd
    import math
    res = model_output

    # Basic objects from the fitted results
    try:
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        conf = res.conf_int()  # DataFrame with index = param names, cols [0,1]
        cov = res.cov_params()
    except Exception as e:
        raise ValueError(f"Provided model_output does not look like a statsmodels results object: {e}")

    param_names = [str(n) for n in params.index]

    # Helper to find parameter name robustly
    def find_main_name():
        # Prefer exact 'reader_view'
        if 'reader_view' in param_names:
            return 'reader_view'
        # fallback: any param that equals or starts with reader_view
        for n in param_names:
            if n == 'reader_view' or n.startswith('reader_view'):
                return n
        # final fallback: any param containing 'reader_view'
        for n in param_names:
            if 'reader_view' in n:
                return n
        return None

    def find_interaction_name():
        # Common expected name is 'reader_view:dyslexia_bin'
        for n in param_names:
            if ':' in n and 'reader_view' in n and 'dyslexia_bin' in n:
                return n
        # If dyslexia_bin was treated as categorical it might appear differently,
        # look for any param that includes both substrings
        for n in param_names:
            if 'reader_view' in n and 'dyslexia' in n:
                return n
        return None

    name_rv = find_main_name()
    name_int = find_interaction_name()

    if name_rv is None:
        raise ValueError("Could not find a parameter corresponding to 'reader_view' in the model parameters.")

    # Extract main effect stats
    coef_rv = float(params[name_rv])
    se_rv = float(bse[name_rv]) if name_rv in bse.index else None
    p_rv = float(pvals[name_rv]) if name_rv in pvals.index else None
    if name_rv in conf.index:
        ci_rv = [float(conf.loc[name_rv, 0]), float(conf.loc[name_rv, 1])]
    else:
        ci_rv = [None, None]

    # Interaction may be absent
    if name_int is not None:
        coef_int = float(params[name_int])
        se_int = float(bse[name_int]) if name_int in bse.index else None
        p_int = float(pvals[name_int]) if name_int in pvals.index else None
        if name_int in conf.index:
            ci_int = [float(conf.loc[name_int, 0]), float(conf.loc[name_int, 1])]
        else:
            ci_int = [None, None]
    else:
        coef_int = 0.0
        se_int = None
        p_int = None
        ci_int = [None, None]

    # Marginal effects on log_speed
    # For dyslexia_bin = 0 (no dyslexia): effect = coef_rv
    eff_no_dys = coef_rv

    # For dyslexia_bin = 1 (has dyslexia): effect = coef_rv + coef_int
    eff_with_dys = coef_rv + coef_int

    # Compute standard errors for the combined effect (if covariance available)
    se_eff_no = None
    se_eff_with = None
    try:
        # variance of coef_rv
        var_rv = float(cov.loc[name_rv, name_rv])
        se_eff_no = math.sqrt(var_rv)
        if name_int is not None:
            var_int = float(cov.loc[name_int, name_int])
            cov_rv_int = float(cov.loc[name_rv, name_int])
            var_sum = var_rv + var_int + 2.0 * cov_rv_int
            se_eff_with = math.sqrt(max(var_sum, 0.0))
        else:
            se_eff_with = se_eff_no
    except Exception:
        # Fallback: combine independent SEs (conservative)
        try:
            se_eff_no = float(se_rv) if se_rv is not None else None
            if se_rv is not None and se_int is not None:
                se_eff_with = float(np.sqrt(se_rv**2 + se_int**2))
            else:
                se_eff_with = se_eff_no
        except Exception:
            se_eff_no = se_eff_no or None
            se_eff_with = se_eff_with or None

    # Compute z and p-values and 95% CIs for marginal effects
    def z_p_ci(effect, se):
        if se is None or se == 0:
            return {"z": None, "p": None, "ci_lower": None, "ci_upper": None}
        z = effect / se
        # two-sided p-value from normal approximation
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se
        return {"z": float(z), "p": float(p), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}

    stats_no = z_p_ci(eff_no_dys, se_eff_no)
    stats_with = z_p_ci(eff_with_dys, se_eff_with)

    # Convert log-scale effects to multiplicative effects and percent change
    def exp_pct(effect):
        mult = math.exp(effect)
        pct = (mult - 1.0) * 100.0
        return {"multiplier": float(mult), "percent_change": float(pct)}

    mult_no = exp_pct(eff_no_dys)
    mult_with = exp_pct(eff_with_dys)

    # Build return object
    results = {
        "parameter_names": {
            "reader_view": name_rv,
            "interaction": name_int
        },
        "reader_view": {
            "coef_log": coef_rv,
            "se": se_rv,
            "p_value": p_rv,
            "conf_int_95": ci_rv,
            "interpretation_log": "Effect of turning Reader View ON (baseline: dyslexia_bin=0)",
            "multiplicative": mult_no["multiplier"],
            "percent_change": mult_no["percent_change"],
        },
        "interaction": {
            "coef_log": coef_int,
            "se": se_int,
            "p_value": p_int,
            "conf_int_95": ci_int,
            "interpretation_log": "Additional effect of Reader View when dyslexia_bin=1 (interaction term)",
        },
        "marginal_effects": {
            "no_dyslexia": {
                "coef_log": eff_no_dys,
                "se": se_eff_no,
                "z": stats_no["z"],
                "p_value": stats_no["p"],
                "conf_int_95": [stats_no["ci_lower"], stats_no["ci_upper"]],
                "multiplier": mult_no["multiplier"],
                "percent_change": mult_no["percent_change"],
                "description": "Estimated multiplicative effect on reading speed when Reader View is ON for readers without dyslexia (dyslexia_bin=0)."
            },
            "with_dyslexia": {
                "coef_log": eff_with_dys,
                "se": se_eff_with,
                "z": stats_with["z"],
                "p_value": stats_with["p"],
                "conf_int_95": [stats_with["ci_lower"], stats_with["ci_upper"]],
                "multiplier": mult_with["multiplier"],
                "percent_change": mult_with["percent_change"],
                "description": "Estimated multiplicative effect on reading speed when Reader View is ON for readers with dyslexia (dyslexia_bin=1)."
            }
        },
        "notes": (
            "Dependent variable is log(speed). Coefficients are additive on the log scale; "
            "exp(coef) gives the multiplicative change in speed. "
            "Marginal effect for dyslexia=1 equals coef(reader_view) + coef(reader_view:dyslexia_bin). "
            "P-values and CIs for the combined effect use the covariance matrix when available."
        )
    }

    # Short human-readable description
    # Determine whether Reader View improves speed for dyslexia group at alpha=0.05
    sig_with = None
    if results["marginal_effects"]["with_dyslexia"]["p_value"] is not None:
        sig_with = results["marginal_effects"]["with_dyslexia"]["p_value"] < 0.05
    sig_no = None
    if results["marginal_effects"]["no_dyslexia"]["p_value"] is not None:
        sig_no = results["marginal_effects"]["no_dyslexia"]["p_value"] < 0.05

    desc_lines = []
    desc_lines.append("Primary quantities extracted:")
    desc_lines.append(f"- Reader View main effect (log scale): {coef_rv:.4f} (SE={se_rv:.4f}, p={p_rv:.4g})" if se_rv is not None else f"- Reader View main effect (log scale): {coef_rv:.4f} (p={p_rv:.4g})")
    if name_int is not None:
        desc_lines.append(f"- Interaction (reader_view x dyslexia): {coef_int:.4f} (SE={se_int:.4f}, p={p_int:.4g})" if se_int is not None else f"- Interaction (reader_view x dyslexia): {coef_int:.4f} (p={p_int:.4g})")
    desc_lines.append(f"- Marginal effect for readers with dyslexia (log): {eff_with_dys:.4f}, corresponds to multiplier={mult_with['multiplier']:.4f} ({mult_with['percent_change']:.2f}% change). " +
                      (f"p={results['marginal_effects']['with_dyslexia']['p_value']:.4g}." if results['marginal_effects']['with_dyslexia']['p_value'] is not None else "p-value not available."))
    desc_lines.append(f"- Marginal effect for readers without dyslexia (log): {eff_no_dys:.4f}, corresponds to multiplier={mult_no['multiplier']:.4f} ({mult_no['percent_change']:.2f}% change). " +
                      (f"p={results['marginal_effects']['no_dyslexia']['p_value']:.4g}." if results['marginal_effects']['no_dyslexia']['p_value'] is not None else "p-value not available."))
    desc_lines.append("")
    # Final yes/no style statement (conservative)
    if sig_with is True:
        desc_lines.append("Conclusion (alpha=0.05): There is evidence that Reader View changes reading speed for individuals with dyslexia.")
        # indicate direction
        if mult_with["percent_change"] > 0:
            desc_lines.append(f"Direction: estimated increase in speed by {mult_with['percent_change']:.2f}% when Reader View is ON for readers with dyslexia.")
        else:
            desc_lines.append(f"Direction: estimated decrease in speed by {abs(mult_with['percent_change']):.2f}% when Reader View is ON for readers with dyslexia.")
    elif sig_with is False:
        desc_lines.append("Conclusion (alpha=0.05): There is no statistically significant evidence that Reader View changes reading speed for individuals with dyslexia.")
        desc_lines.append(f"Estimated effect: {mult_with['percent_change']:.2f}% change (not statistically significant).")
    else:
        desc_lines.append("Conclusion: Could not determine statistical significance for the dyslexia subgroup (p-value not available).")

    description = "\n".join(desc_lines)

    return {"object": results, "description": description}