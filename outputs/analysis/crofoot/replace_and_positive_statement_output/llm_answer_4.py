def extract_final_answer(model_output):
    """
    Extracts coefficients, (clustered) standard errors, p-values, 95% CIs, and odds ratios
    for the predictors of interest from the provided model_output dictionary.

    Returns a dictionary with keys:
      - "object": dict keyed by parameter name with numeric summaries (coef, se, p, CI, OR, OR_CI)
      - "description": short plain-language interpretation of the results in context
    """
    import numpy as np
    from math import exp

    # Required keys in model_output
    if 'model_result' not in model_output:
        raise ValueError("model_output must contain key 'model_result'")

    res = model_output['model_result']
    params = res.params  # pandas Series
    param_names = list(params.index)

    # Try to use clustered SEs and clustered p-values if provided
    clustered_se = model_output.get('clustered_se', None)
    clustered_pvals = model_output.get('clustered_pvalues', None)
    clustered_cov = model_output.get('clustered_cov', None)

    # Fallback to model-provided SEs and p-values if clustered are missing
    model_se = getattr(res, 'bse', None)
    model_pvals = getattr(res, 'pvalues', None)
    conf_int_df = None
    try:
        conf_int_df = res.conf_int()
    except Exception:
        conf_int_df = None

    # Helper to get numeric summary for a parameter index
    def summarize_param(idx):
        name = param_names[idx]
        coef = float(params.iloc[idx])
        # choose se and p
        if clustered_se is not None:
            try:
                se = float(clustered_se[idx])
            except Exception:
                se = None
        elif model_se is not None:
            try:
                se = float(model_se.iloc[idx])
            except Exception:
                se = None
        else:
            se = None

        if clustered_pvals is not None:
            try:
                p = float(clustered_pvals[idx])
            except Exception:
                p = None
        elif model_pvals is not None:
            try:
                p = float(model_pvals.iloc[idx])
            except Exception:
                p = None
        else:
            p = None

        # 95% CI: prefer clustered se if available, otherwise use res.conf_int()
        if se is not None:
            z = 1.96
            ci_lower = coef - z * se
            ci_upper = coef + z * se
        elif conf_int_df is not None:
            try:
                ci_lower = float(conf_int_df.iloc[idx, 0])
                ci_upper = float(conf_int_df.iloc[idx, 1])
            except Exception:
                ci_lower = ci_upper = None
        else:
            ci_lower = ci_upper = None

        # Odds ratio and its CI (on exponentiated scale)
        try:
            or_val = exp(coef)
            or_ci_lower = exp(ci_lower) if ci_lower is not None else None
            or_ci_upper = exp(ci_upper) if ci_upper is not None else None
        except Exception:
            or_val = or_ci_lower = or_ci_upper = None

        return {
            'coef': coef,
            'se': se,
            'p_value': p,
            'ci_95': (ci_lower, ci_upper),
            'odds_ratio': or_val,
            'odds_ratio_ci_95': (or_ci_lower, or_ci_upper),
        }

    # Identify indices for predictors of interest
    # Expected parameter order (typical for statsmodels with the formula used):
    # ['Intercept', 'SizeDiff_z', 'LocAdv_z', 'SizeDiff_z:LocAdv_z', 'm_diff_z', 'f_diff_z']
    # But we'll find indices by name robustly.
    def find_index_by_name(target):
        for i, name in enumerate(param_names):
            if name == target:
                return i
        return None

    idx_size = find_index_by_name('SizeDiff_z')
    idx_loc = find_index_by_name('LocAdv_z')
    # interaction name may be 'SizeDiff_z:LocAdv_z' or 'SizeDiff_z*LocAdv_z' depending on model; statsmodels uses 'SizeDiff_z:LocAdv_z'
    idx_inter = find_index_by_name('SizeDiff_z:LocAdv_z')

    results = {}
    missing = []
    for label, idx in (('SizeDiff_z', idx_size), ('LocAdv_z', idx_loc), ('SizeDiff_z:LocAdv_z', idx_inter)):
        if idx is None:
            missing.append(label)
        else:
            results[label] = summarize_param(idx)

    # Attempt to extract average marginal effects if available
    ame_summary = None
    marg_obj = model_output.get('marginal_effects', None)
    if marg_obj is not None:
        try:
            # statsmodels DiscreteMargins usually supports summary_frame()
            sf = marg_obj.summary_frame()
            # convert to dict for convenience
            ame_summary = sf.to_dict(orient='index')
        except Exception:
            try:
                # fallback: try .margeff or .summary()
                ame_summary = str(marg_obj)
            except Exception:
                ame_summary = None

    # Build plain-language description
    desc_lines = []
    if missing:
        desc_lines.append(f"Warning: could not find parameter(s) in model output: {missing}.")
    desc_lines.append("Summary for focal-group predictors (coefficients are on the log-odds scale; odds ratios are exp(coef)):")

    for key in ['SizeDiff_z', 'LocAdv_z', 'SizeDiff_z:LocAdv_z']:
        if key in results:
            r = results[key]
            coef = r['coef']
            se = r['se']
            p = r['p_value']
            ci = r['ci_95']
            orv = r['odds_ratio']
            orci = r['odds_ratio_ci_95']
            sig_text = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"

            coef_str = f"{coef:.3f}"
            se_str = f"{se:.3f}" if se is not None else "NA"
            p_str = f"{p:.3f}" if p is not None else "NA"

            if ci[0] is not None and ci[1] is not None:
                ci_str = f"95% CI (log-odds)=({ci[0]:.3f}, {ci[1]:.3f})"
                desc_lines.append(f"- {key}: coef={coef_str}, se={se_str}, p={p_str}; {ci_str}")
            else:
                desc_lines.append(f"- {key}: coef={coef_str}, se={se_str}, p={p_str}; 95% CI unavailable.")

            # Add odds ratio line
            if orv is not None and orci[0] is not None and orci[1] is not None:
                desc_lines.append(f"  odds ratio={orv:.3f}, 95% CI=({orci[0]:.3f}, {orci[1]:.3f}) — {sig_text}.")
            else:
                desc_lines.append(f"  odds ratio unavailable — {sig_text}.")
        else:
            desc_lines.append(f"- {key}: parameter not found in model results.")

    # Overall interpretation
    # Decide significance based on clustered p-values if available, else model p-values
    def is_significant(p):
        return (p is not None) and (p < 0.05)

    size_sig = results.get('SizeDiff_z', {}).get('p_value') if 'SizeDiff_z' in results else None
    loc_sig = results.get('LocAdv_z', {}).get('p_value') if 'LocAdv_z' in results else None
    inter_sig = results.get('SizeDiff_z:LocAdv_z', {}).get('p_value') if 'SizeDiff_z:LocAdv_z' in results else None

    if any(is_significant(p) for p in (size_sig, loc_sig, inter_sig)):
        desc_lines.append("At least one focal predictor shows a statistically significant association with winning (see parameter lines above).")
    else:
        desc_lines.append("No evidence (p >= 0.05 using the available standard errors) that relative group size, contest location, or their interaction significantly influence the probability that the focal group wins.")

    if ame_summary is not None:
        desc_lines.append("Average marginal effects were also computed (present in returned object under 'marginal_effects').")

    description = " ".join(desc_lines)

    # Final object returned: include numeric summaries and the marginal effects object/frame if available
    final_object = {
        'parameter_summaries': results,
        'marginal_effects_summary': ame_summary,
        'aic': model_output.get('aic'),
        'bic': model_output.get('bic'),
    }

    return {
        'object': final_object,
        'description': description
    }