def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether 'Reader View' improves reading speed,
    especially for readers with dyslexia.

    Returns a dictionary with keys:
      - "object": a dict containing estimates, SEs, p-values, and 95% CIs for:
          * ReaderView effect for non-dyslexic readers (main effect)
          * ReaderView effect for dyslexic readers (main effect + interaction)
          * Interaction term (ReaderView:Dyslexia) if present
      - "description": short plain-language interpretation of those statistics
                     addressing whether Reader View improves reading speed for
                     dyslexic readers.

    The function expects a fitted statsmodels results object (e.g., MixedLMResults).
    """
    import numpy as np
    from math import erf, sqrt

    # Helper: normal two-sided p-value from z
    def normal_pvalue(z):
        # normal cdf via erf to avoid SciPy dependency if not available
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return 2.0 * (1.0 - cdf) if z >= 0 else 2.0 * cdf

    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided (model_output is None). Cannot extract statistics."
        }

    # Check that object has the expected attributes
    required_attrs = ['params', 'bse', 'cov_params', 'conf_int']
    for a in required_attrs:
        if not hasattr(model_output, a):
            return {
                "object": None,
                "description": f"The provided model_output does not appear to be a fitted statsmodels results object (missing '{a}')."
            }

    params = model_output.params
    bse = model_output.bse
    pvalues = getattr(model_output, 'pvalues', None)  # pvalues may exist
    try:
        cov = model_output.cov_params()
    except Exception:
        # cov_params might be a method or attribute; try calling if callable
        cov = model_output.cov_params if not callable(model_output.cov_params) else model_output.cov_params()

    ci_df = model_output.conf_int()

    # Names used in the formula
    main_name = 'ReaderView'
    interact_name = 'ReaderView:Dyslexia'

    # Ensure we have an indexable params (Series)
    param_index = list(params.index)

    results = {}

    # Extract main effect (ReaderView) if present
    if main_name in param_index:
        coef_main = float(params[main_name])
        se_main = float(bse[main_name]) if main_name in bse.index else float(np.nan)
        p_main = float(pvalues[main_name]) if (pvalues is not None and main_name in pvalues.index) else float(np.nan)
        try:
            ci_main = [float(ci_df.loc[main_name, 0]), float(ci_df.loc[main_name, 1])]
        except Exception:
            ci_main = [coef_main - 1.96 * se_main, coef_main + 1.96 * se_main]
        results['ReaderView_non_dyslexic'] = {
            "estimate_wpm": coef_main,
            "se": se_main,
            "p_value": p_main,
            "ci_95_wpm": ci_main,
            "interpretation": "Effect of Reader View for non-dyslexic readers (Dyslexia=0)."
        }
    else:
        results['ReaderView_non_dyslexic'] = None

    # Extract interaction term if present
    interaction_present = interact_name in param_index
    if interaction_present:
        coef_inter = float(params[interact_name])
        se_inter = float(bse[interact_name]) if interact_name in bse.index else float(np.nan)
        p_inter = float(pvalues[interact_name]) if (pvalues is not None and interact_name in pvalues.index) else float(np.nan)
        try:
            ci_inter = [float(ci_df.loc[interact_name, 0]), float(ci_df.loc[interact_name, 1])]
        except Exception:
            ci_inter = [coef_inter - 1.96 * se_inter, coef_inter + 1.96 * se_inter]
        results['ReaderView_by_Dyslexia_interaction'] = {
            "estimate_wpm": coef_inter,
            "se": se_inter,
            "p_value": p_inter,
            "ci_95_wpm": ci_inter,
            "interpretation": "How the Reader View effect differs when Dyslexia=1 vs Dyslexia=0."
        }
    else:
        results['ReaderView_by_Dyslexia_interaction'] = None

    # Compute combined effect for dyslexic readers: ReaderView + ReaderView:Dyslexia
    if main_name in param_index:
        if interaction_present:
            # build contrast vector a: 1 for main_name, 1 for interaction, 0 otherwise
            idx_map = {name: i for i, name in enumerate(param_index)}
            a = np.zeros(len(param_index))
            a[idx_map[main_name]] = 1.0
            a[idx_map[interact_name]] = 1.0
            est = float(np.dot(a, params.values))
            # cov may be DataFrame; ensure consistent ordering
            if hasattr(cov, 'loc'):
                cov_mat = cov.loc[param_index, param_index].values
            else:
                cov_mat = np.asarray(cov)
            var = float(np.dot(a, np.dot(cov_mat, a)))
            se = float(np.sqrt(var)) if var >= 0 else float(np.nan)
            z = est / se if se and not np.isnan(se) else float('nan')
            p_combined = normal_pvalue(abs(z)) if not np.isnan(z) else float(np.nan)
            ci_combined = [est - 1.96 * se, est + 1.96 * se] if not np.isnan(se) else [float(np.nan), float(np.nan)]
            results['ReaderView_dyslexic'] = {
                "estimate_wpm": est,
                "se": se,
                "z_value": z,
                "p_value": p_combined,
                "ci_95_wpm": ci_combined,
                "interpretation": "Estimated effect of Reader View for dyslexic readers (Dyslexia=1): ReaderView + ReaderView:Dyslexia."
            }
        else:
            # No interaction term: effect for dyslexic readers equals main effect
            results['ReaderView_dyslexic'] = results['ReaderView_non_dyslexic']
    else:
        results['ReaderView_dyslexic'] = None

    # Short plain-language description / conclusion
    desc_lines = []
    if results['ReaderView_dyslexic'] is None:
        desc_lines.append("Could not compute Reader View effect for dyslexic readers (missing parameters).")
    else:
        r = results['ReaderView_dyslexic']
        est = r['estimate_wpm']
        pval = r['p_value']
        if np.isnan(est):
            desc_lines.append("Computed effect is NaN; something went wrong with variance/covariance matrix.")
        else:
            sign = "increase" if est > 0 else ("decrease" if est < 0 else "no change")
            # Decide statistical significance using p < 0.05 if available
            if not np.isnan(pval):
                sig_text = "statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)"
                desc_lines.append(f"For dyslexic readers, activating Reader View changes reading speed by {est:.2f} WPM on average ({sign}); this effect is {sig_text} (p = {pval:.3g}).")
            else:
                desc_lines.append(f"For dyslexic readers, activating Reader View changes reading speed by {est:.2f} WPM on average ({sign}); p-value not available.")

    # Also comment on whether effect differs between groups (interaction)
    if results['ReaderView_by_Dyslexia_interaction'] is None:
        desc_lines.append("No interaction term present in the model, so no evidence that the Reader View effect differs by dyslexia status was estimated separately.")
    else:
        inter = results['ReaderView_by_Dyslexia_interaction']
        if not np.isnan(inter['p_value']):
            if inter['p_value'] < 0.05:
                desc_lines.append(f"The interaction term (ReaderView:Dyslexia) is statistically significant (p = {inter['p_value']:.3g}), indicating the Reader View effect differs between dyslexic and non-dyslexic readers.")
            else:
                desc_lines.append(f"The interaction term (ReaderView:Dyslexia) is not statistically significant (p = {inter['p_value']:.3g}), providing no evidence that the Reader View effect differs by dyslexia status.")
        else:
            desc_lines.append("Interaction term present but p-value not available.")

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }