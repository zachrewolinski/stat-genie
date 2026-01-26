def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and odds ratios
    for the key predictors in the fitted GLM results object returned by the provided model().
    
    Returns a dictionary with:
      - "object": dict containing a coefficient table for relevant terms and the derived
                  effect of relative group size when the contest is near the other group's
                  territory (LocationFocal = 0) vs near/in the focal group's territory
                  (LocationFocal = 1). Each entry contains coef, se, p, 95% CI, OR, OR CI.
      - "description": plain-language explanation of what the numbers mean.
    """
    import numpy as np
    from math import exp
    try:
        # Prefer pandas-like indexing if available
        params = model_output.params
    except Exception as e:
        raise ValueError("Could not access params from model_output: " + str(e))
    
    # cov matrix (robust if returned by get_robustcov_results)
    try:
        cov = model_output.cov_params()
    except Exception:
        # some objects store cov_params as attribute
        cov = getattr(model_output, "cov_params", None)
        if cov is None:
            raise ValueError("Could not access covariance matrix from model_output.")
    
    # ensure cov is a DataFrame-like with indices matching params
    try:
        import pandas as pd
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        # fallback: try to construct index from params
        try:
            cov = np.asarray(cov)
        except Exception:
            raise ValueError("Covariance matrix has unexpected format.")
    
    # obtain standard errors and p-values (compute p-values if missing)
    try:
        bse = model_output.bse
    except Exception:
        # compute from covariance diagonal
        bse = np.sqrt(np.diag(cov))
    
    try:
        pvalues = model_output.pvalues
    except Exception:
        # compute normal-approx p-values using z = coef / se
        from scipy import stats
        z = params / bse
        pvalues = 2 * stats.norm.sf(np.abs(z))
    
    # helper to find parameter name (exact or containing substring)
    def find_name(substr):
        # substr should be one of the column variable names from model design: 
        # 'size_ratio_z','LocationFocal','interaction'
        for name in params.index:
            if name == substr:
                return name
        # fallback: find by containment
        for name in params.index:
            if substr in name:
                return name
        raise KeyError(f"Parameter matching '{substr}' not found in model parameters: {list(params.index)}")
    
    # keys we care about
    try:
        name_size = find_name('size_ratio_z')
        name_loc = find_name('LocationFocal')
        name_inter = find_name('interaction')
    except KeyError as e:
        raise
    
    # extract coefficients etc for the main terms (and optionally controls)
    def summarize_param(name):
        coef = float(params[name])
        se = float(bse[name]) if hasattr(bse, '__getitem__') else float(bse[params.index.get_loc(name)])
        # p-value
        pv = float(pvalues[name]) if hasattr(pvalues, '__getitem__') else float(pvalues[params.index.get_loc(name)])
        # 95% CI (normal approx)
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        # odds ratio and CI
        or_ = float(np.exp(coef))
        or_lo = float(np.exp(lo))
        or_hi = float(np.exp(hi))
        return {
            "coef": coef,
            "se": se,
            "p": pv,
            "ci_lower": lo,
            "ci_upper": hi,
            "odds_ratio": or_,
            "odds_ratio_ci_lower": or_lo,
            "odds_ratio_ci_upper": or_hi
        }
    
    coef_table = {}
    # main predictors
    for n in [name_size, name_loc, name_inter]:
        coef_table[n] = summarize_param(n)
    # include controls if present
    for ctrl in ['male_diff_z', 'female_diff_z']:
        try:
            name_ctrl = find_name(ctrl)
            coef_table[name_ctrl] = summarize_param(name_ctrl)
        except KeyError:
            pass  # control not present (shouldn't happen with provided model, but safe)
    
    # Derived effects: effect of size_ratio_z when LocationFocal = 0 and = 1
    # When LocationFocal = 0: effect = beta_size
    beta_size = float(params[name_size])
    se_size = float(bse[name_size])
    # When LocationFocal = 1: effect = beta_size + beta_interaction
    beta_inter = float(params[name_inter])
    # variance for sum
    var_size = float(cov.loc[name_size, name_size])
    var_inter = float(cov.loc[name_inter, name_inter])
    cov_si = float(cov.loc[name_size, name_inter])
    eff_loc0 = beta_size
    se_loc0 = np.sqrt(var_size)
    eff_loc1 = beta_size + beta_inter
    var_loc1 = var_size + var_inter + 2 * cov_si
    # guard against small negative variance due to numerical issues
    if var_loc1 < 0 and var_loc1 > -1e-12:
        var_loc1 = 0.0
    if var_loc1 < 0:
        raise ValueError(f"Computed negative variance for effect when LocationFocal=1: {var_loc1}")
    se_loc1 = np.sqrt(var_loc1)
    
    # p-values for these linear combinations (normal approx)
    from scipy import stats
    z_loc0 = eff_loc0 / se_loc0 if se_loc0 > 0 else np.nan
    p_loc0 = float(2 * stats.norm.sf(abs(z_loc0))) if se_loc0 > 0 else np.nan
    z_loc1 = eff_loc1 / se_loc1 if se_loc1 > 0 else np.nan
    p_loc1 = float(2 * stats.norm.sf(abs(z_loc1))) if se_loc1 > 0 else np.nan
    
    # CIs and ORs
    ci0_lo = eff_loc0 - 1.96 * se_loc0
    ci0_hi = eff_loc0 + 1.96 * se_loc0
    ci1_lo = eff_loc1 - 1.96 * se_loc1
    ci1_hi = eff_loc1 + 1.96 * se_loc1
    
    derived = {
        "size_effect_when_other_territory (LocationFocal=0)": {
            "log_odds_coef": float(eff_loc0),
            "se": float(se_loc0),
            "p": p_loc0,
            "ci_lower": float(ci0_lo),
            "ci_upper": float(ci0_hi),
            "odds_ratio": float(np.exp(eff_loc0)),
            "odds_ratio_ci_lower": float(np.exp(ci0_lo)),
            "odds_ratio_ci_upper": float(np.exp(ci0_hi))
        },
        "size_effect_when_focal_territory (LocationFocal=1)": {
            "log_odds_coef": float(eff_loc1),
            "se": float(se_loc1),
            "p": p_loc1,
            "ci_lower": float(ci1_lo),
            "ci_upper": float(ci1_hi),
            "odds_ratio": float(np.exp(eff_loc1)),
            "odds_ratio_ci_lower": float(np.exp(ci1_lo)),
            "odds_ratio_ci_upper": float(np.exp(ci1_hi))
        },
        "interpretation_note": (
            "The 'log_odds_coef' gives the change in log-odds of the focal group winning per one "
            "standard-deviation increase in relative group size. The odds ratio is exp(log_odds_coef). "
            "When LocationFocal=0, the coefficient equals beta_size. When LocationFocal=1, the "
            "coefficient equals beta_size + beta_interaction (so the interaction term captures the "
            "additional change in the effect of size when contests occur in/near the focal group's territory)."
        )
    }
    
    result_object = {
        "coef_table": coef_table,
        "derived_size_effects": derived
    }
    
    # Plain-language description
    description_lines = []
    description_lines.append(
        "Extracted coefficients, standard errors, p-values, 95% confidence intervals, and odds ratios "
        "for the model terms of interest."
    )
    description_lines.append(
        "Key parameters and how to interpret them:\n"
        "- 'size_ratio_z' (beta_size): effect of relative group size (in SD units) on the log-odds of the focal group winning when the contest is NOT in/near the focal group's territory (LocationFocal=0).\n"
        "- 'interaction' (size_ratio_z * LocationFocal): additional change in the size effect when the contest is in/near the focal group's territory. The total effect of size in focal territory is beta_size + beta_interaction.\n"
        "- 'LocationFocal' main effect: baseline change in log-odds of focal winning when contest is in/near focal territory (when size_ratio_z = 0).\n"
    )
    description_lines.append(
        "Use the 'derived_size_effects' entries to see the estimated effect (log-odds and odds ratio) "
        "of a one-SD increase in relative group size separately for contests in the other group's territory "
        "(LocationFocal=0) and in/near the focal group's territory (LocationFocal=1), together with p-values "
        "and 95% CIs. If the p-value for the interaction is < 0.05, that suggests the effect of relative size "
        "differs significantly between locations."
    )
    
    description = "\n".join(description_lines)
    
    return {"object": result_object, "description": description}