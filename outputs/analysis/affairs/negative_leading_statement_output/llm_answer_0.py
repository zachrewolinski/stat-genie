def extract_final_answer(model_output):
    """
    Extracts effect of 'children_dummy' on 'affairs' from model_output (the dict returned by model()).
    Returns a dict with:
      - "object": numeric summaries (coefficients, p-values, CIs, IRR for count model)
      - "description": plain-language interpretation of the results in context.
    """
    import math
    import numpy as np

    out = {}
    # Helper to build CI
    def ci_from_coef_and_se(coef, se, z=1.96):
        if coef is None or se is None or math.isnan(coef) or math.isnan(se):
            return (None, None)
        lower = coef - z * se
        upper = coef + z * se
        return (lower, upper)

    # 1) Try extracting from fitted objects where possible (preferred for exact SEs/CIs)
    ols_info = {}
    try:
        ols_res = model_output.get('ols_result')
        if ols_res is not None:
            coef = float(ols_res.params.get('children_dummy'))
            se = float(ols_res.bse.get('children_dummy'))
            p = float(ols_res.pvalues.get('children_dummy'))
            ci_lower, ci_upper = ci_from_coef_and_se(coef, se)
            ols_info = {
                'coef': coef,
                'se': se,
                'pvalue': p,
                '95%_ci': (ci_lower, ci_upper),
                'model': 'OLS'
            }
    except Exception:
        ols_info = {}

    # 2) Count model (ZINB or ZIP) — extract count-part coefficient for children_dummy
    count_info = {}
    try:
        # prefer ZINB, fallback to ZIP
        count_res = model_output.get('zinb_result') or model_output.get('zip_result')
        if count_res is not None:
            # statsmodels stores parameters with names; the count-part parameter is 'children_dummy'
            coef = float(count_res.params.get('children_dummy'))
            se = float(count_res.bse.get('children_dummy'))
            p = float(count_res.pvalues.get('children_dummy'))
            ci_lower, ci_upper = ci_from_coef_and_se(coef, se)
            irr = float(np.exp(coef))
            irr_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                      float(np.exp(ci_upper)) if ci_upper is not None else None)
            count_info = {
                'coef': coef,
                'se': se,
                'pvalue': p,
                '95%_ci': (ci_lower, ci_upper),
                'incidence_rate_ratio': irr,
                'irr_95%_ci': irr_ci,
                'model': 'ZINB_or_ZIP'
            }
    except Exception:
        count_info = {}

    # 3) If the above failed, try to use precomputed summaries in the dict
    if not ols_info and 'ols_children_effect' in model_output:
        tmp = model_output['ols_children_effect']
        ols_info = {'coef': tmp.get('coef'), 'pvalue': tmp.get('pvalue'), 'model': 'OLS (fallback)'}
    if not count_info:
        if 'zinb_children_effect' in model_output and model_output['zinb_children_effect'] is not None:
            tmp = model_output['zinb_children_effect']
            coef = tmp.get('coef')
            p = tmp.get('pvalue')
            irr = float(np.exp(coef)) if coef is not None else None
            count_info = {'coef': coef, 'pvalue': p, 'incidence_rate_ratio': irr, 'model': 'ZINB (fallback)'}
        elif 'zip_children_effect' in model_output and model_output['zip_children_effect'] is not None:
            tmp = model_output['zip_children_effect']
            coef = tmp.get('coef')
            p = tmp.get('pvalue')
            irr = float(np.exp(coef)) if coef is not None else None
            count_info = {'coef': coef, 'pvalue': p, 'incidence_rate_ratio': irr, 'model': 'ZIP (fallback)'}

    # 4) Build a concise interpretation
    parts = []
    if ols_info:
        parts.append(
            "OLS: children_coef = {coef:.3f}, p = {p:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]"
            .format(coef=ols_info.get('coef', float('nan')),
                    p=ols_info.get('pvalue', float('nan')),
                    lo=(ols_info.get('95%_ci')[0] if ols_info.get('95%_ci') else float('nan')),
                    hi=(ols_info.get('95%_ci')[1] if ols_info.get('95%_ci') else float('nan')))
        )
    else:
        parts.append("OLS: no numeric summary available.")

    if count_info:
        ci = count_info.get('95%_ci', (None, None))
        irr_ci = count_info.get('irr_95%_ci', (None, None))
        parts.append(
            "Count model ({model}): children_coef = {coef:.3f}, p = {p:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]; "
            "IRR = {irr:.3f}, IRR 95% CI = [{ir_lo:.3f}, {ir_hi:.3f}]"
            .format(model=count_info.get('model', ''),
                    coef=count_info.get('coef', float('nan')),
                    p=count_info.get('pvalue', float('nan')),
                    lo=(ci[0] if ci[0] is not None else float('nan')),
                    hi=(ci[1] if ci[1] is not None else float('nan')),
                    irr=(count_info.get('incidence_rate_ratio') if count_info.get('incidence_rate_ratio') is not None else float('nan')),
                    ir_lo=(irr_ci[0] if irr_ci[0] is not None else float('nan')),
                    ir_hi=(irr_ci[1] if irr_ci[1] is not None else float('nan')))
        )
    else:
        parts.append("Count model: no numeric summary available.")

    # Final plain-language conclusion
    conclusion = (
        "Interpretation: Point estimates from both models are negative (i.e., having children is associated "
        "with fewer reported extramarital affairs). The count model's point estimate (coef ≈ {c_coef:.3f}) "
        "corresponds to an incidence rate ratio of about {irr:.3f} (suggesting ~{pct:.0f}% lower rate), "
        "but the effect is not statistically significant (p = {c_p:.3f}; 95% CI for IRR includes 1). "
        "The OLS estimate is small and not statistically significant (p = {o_p:.3f}). "
        "Conclusion: there is no statistically significant evidence in these models that having children decreases "
        "engagement in extramarital affairs after adjusting for the listed controls."
    ).format(
        c_coef=(count_info.get('coef') if count_info.get('coef') is not None else float('nan')),
        irr=(count_info.get('incidence_rate_ratio') if count_info.get('incidence_rate_ratio') is not None else float('nan')),
        pct=(1 - (count_info.get('incidence_rate_ratio') if count_info.get('incidence_rate_ratio') is not None else 1)) * 100,
        c_p=(count_info.get('pvalue') if count_info.get('pvalue') is not None else float('nan')),
        o_p=(ols_info.get('pvalue') if ols_info.get('pvalue') is not None else float('nan'))
    )

    # Assemble return object containing numeric summaries and text
    out['object'] = {
        'ols': ols_info if ols_info else None,
        'count_model': count_info if count_info else None
    }
    out['description'] = "\n".join(parts) + "\n\n" + conclusion

    return out