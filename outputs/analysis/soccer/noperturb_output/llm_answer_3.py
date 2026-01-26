def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of is_dark on redCards from a fitted statsmodels results object.

    Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, se, p-value, 95% CI, IRR and IRR CI;
                  interaction term info and simple effects at mean +/- 1 SD of meanIAT when available)
      - "description": short plain-language interpretation focused on whether dark-skinned players
                       are more likely to receive red cards than light-skinned players.
    """
    import numpy as np
    from math import sqrt
    from scipy.stats import norm

    res = model_output

    # Try to access expected result attributes
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        ci_df = res.conf_int()
        cov = res.cov_params()
    except Exception as e:
        raise ValueError("Unable to read expected attributes from model_output. "
                         "Ensure this is a statsmodels results object (possibly robustified). "
                         f"Underlying error: {e}")

    param_names = list(params.index)

    # Find the main effect parameter for is_dark
    # Prefer exact 'is_dark', otherwise take any parameter containing 'is_dark' but not an interaction
    main_candidates = [n for n in param_names if n == 'is_dark' or ('is_dark' in n and ':' not in n)]
    if not main_candidates:
        raise ValueError("Could not find a main-effect parameter for 'is_dark' in model parameters: "
                         f"{param_names}")
    main_name = main_candidates[0]

    coef = float(params[main_name])
    se = float(bse[main_name]) if main_name in bse.index else None
    pval = float(pvalues[main_name]) if main_name in pvalues.index else None
    # Confidence interval: statsmodels returns DataFrame with two columns
    if main_name in ci_df.index:
        ci_lower, ci_upper = float(ci_df.loc[main_name].iloc[0]), float(ci_df.loc[main_name].iloc[1])
    else:
        ci_lower, ci_upper = None, None

    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    results_dict = {
        'main_parameter_name': main_name,
        'coef': coef,
        'se': se,
        'pvalue': pval,
        'conf_int_95': (ci_lower, ci_upper),
        'incidence_rate_ratio': irr,
        'irr_95_ci': (irr_ci_lower, irr_ci_upper),
    }

    # Look for interaction term is_dark:meanIAT (or meanIAT:is_dark)
    interaction_candidates = [n for n in param_names if 'is_dark' in n and ':' in n]
    interaction_info = None
    simple_effects = None

    if interaction_candidates:
        inter_name = interaction_candidates[0]
        inter_coef = float(params[inter_name])
        inter_se = float(bse[inter_name]) if inter_name in bse.index else None
        inter_p = float(pvalues[inter_name]) if inter_name in pvalues.index else None
        if inter_name in ci_df.index:
            inter_ci_lower, inter_ci_upper = float(ci_df.loc[inter_name].iloc[0]), float(ci_df.loc[inter_name].iloc[1])
        else:
            inter_ci_lower, inter_ci_upper = None, None

        interaction_info = {
            'interaction_parameter_name': inter_name,
            'coef': inter_coef,
            'se': inter_se,
            'pvalue': inter_p,
            'conf_int_95': (inter_ci_lower, inter_ci_upper),
        }

        # If data frame is available through the model, compute simple effects of is_dark at
        # mean(meanIAT) and mean +/- 1 SD. This uses covariance matrix to get SEs.
        try:
            df = None
            if hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'frame'):
                df = res.model.data.frame
            elif hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'orig_endog'):
                # fallback - less likely to be useful
                df = getattr(res.model.data, 'frame', None)

            if df is not None and 'meanIAT' in df.columns:
                meanIAT = float(df['meanIAT'].mean())
                sdIAT = float(df['meanIAT'].std(ddof=0))  # population sd to get +/-1 SD
                vals = {
                    'mean': meanIAT,
                    'mean_minus_1sd': meanIAT - sdIAT,
                    'mean_plus_1sd': meanIAT + sdIAT
                }

                # covariance matrix: try to access robust cov (should reflect clustering if used)
                cov_mat = cov
                simple_effects = {}
                for label, v in vals.items():
                    # combined coef = beta_is_dark + v * beta_interaction
                    combined_beta = coef + v * inter_coef
                    # Var(combined) = Var(b1) + v^2 Var(b2) + 2 v Cov(b1,b2)
                    try:
                        var_b1 = float(cov_mat.loc[main_name, main_name])
                        var_b2 = float(cov_mat.loc[inter_name, inter_name])
                        cov_b1b2 = float(cov_mat.loc[main_name, inter_name])
                        var_comb = var_b1 + (v**2) * var_b2 + 2 * v * cov_b1b2
                        se_comb = sqrt(max(var_comb, 0.0))
                        z = combined_beta / se_comb if se_comb > 0 else np.nan
                        p_comb = 2 * (1 - norm.cdf(abs(z))) if se_comb > 0 else None
                        ci_low = combined_beta - 1.96 * se_comb
                        ci_high = combined_beta + 1.96 * se_comb
                        irr_comb = float(np.exp(combined_beta))
                        irr_ci_low = float(np.exp(ci_low))
                        irr_ci_high = float(np.exp(ci_high))
                    except Exception:
                        se_comb = None
                        p_comb = None
                        ci_low = ci_high = irr_comb = irr_ci_low = irr_ci_high = None

                    simple_effects[label] = {
                        'meanIAT_value': v,
                        'combined_coef_is_dark_at_value': combined_beta,
                        'se': se_comb,
                        'pvalue': p_comb,
                        'conf_int_95': (ci_low, ci_high),
                        'incidence_rate_ratio': irr_comb,
                        'irr_95_ci': (irr_ci_low, irr_ci_high)
                    }
            else:
                # No meanIAT data available: skip simple effects
                simple_effects = None
        except Exception:
            # If any step fails, skip simple effects but keep interaction coefficients
            simple_effects = None

    # Populate results
    results_dict['interaction'] = interaction_info
    results_dict['simple_effects_of_is_dark_at_meanIAT'] = simple_effects

    # Short interpretation focused on the yes/no question
    # Decision rule: if IRR > 1 and p < 0.05 we conclude "more likely"; otherwise "no strong evidence".
    interp = []
    if pval is None:
        interp.append("Could not determine p-value for the is_dark coefficient.")
    else:
        if irr > 1 and pval < 0.05:
            interp.append(f"The model's main effect for {main_name} is positive (coef={coef:.3f}, IRR={irr:.3f}) "
                          f"and statistically significant (p={pval:.3f}). This indicates dark-skinned players "
                          "receive more red cards than light-skinned players, controlling for covariates.")
        elif irr > 1 and pval >= 0.05:
            interp.append(f"The coefficient suggests a higher rate for dark-skinned players (IRR={irr:.3f}), "
                          f"but this effect is not statistically significant (p={pval:.3f}); therefore there is "
                          "no strong evidence to conclude a difference.")
        elif irr < 1 and pval < 0.05:
            interp.append(f"The model's main effect indicates dark-skinned players receive fewer red cards "
                          f"(IRR={irr:.3f}), and the effect is statistically significant (p={pval:.3f}).")
        else:
            interp.append(f"The model does not provide evidence that dark-skinned players receive more red cards "
                          f"(coef={coef:.3f}, IRR={irr:.3f}, p={pval:.3f}).")

    # If there is an interaction and simple effects were computed, add a short note
    if interaction_info is not None:
        if simple_effects is not None:
            interp.append("An interaction with meanIAT was estimated; simple effects of is_dark at mean +/-1 SD of meanIAT "
                          "are provided in the returned 'object'. Interpret those to see whether the effect varies by referee implicit bias.")
        else:
            interp.append("An interaction between is_dark and meanIAT was estimated (see returned 'object'), "
                          "but simple-effect estimates at representative meanIAT values were not computed (data unavailable).")

    description = " ".join(interp)

    return {"object": results_dict, "description": description}