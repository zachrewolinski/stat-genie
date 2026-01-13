def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, odds ratios, and 95% confidence intervals
    for the main predictor 'Gender' and the interaction 'Gender_Black' from a
    fitted statsmodels Logit result (BinaryResultsWrapper).

    Returns a dictionary with:
      - "object": a dict containing numeric results for 'Gender' and 'Gender_Black'
      - "description": a brief plain-language interpretation of those results

    The function will try to return both standard ML-based statistics and
    robust (HC3) statistics if available.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    def safe_get(df_like, name, default=np.nan):
        try:
            return df_like[name]
        except Exception:
            return default

    # Parameters we care about
    terms = ['Gender', 'Gender_Black']

    # Basic (default) estimates
    params = getattr(res, 'params', pd.Series(dtype=float))
    pvalues = getattr(res, 'pvalues', pd.Series(dtype=float))
    bse = getattr(res, 'bse', pd.Series(dtype=float))
    conf_int = None
    try:
        conf_int = res.conf_int()
    except Exception:
        conf_int = None

    # Robust (HC3) estimates if available
    robust_res = None
    try:
        robust_res = res.get_robustcov_results(cov_type='HC3')
        robust_params = getattr(robust_res, 'params', pd.Series(dtype=float))
        robust_pvalues = getattr(robust_res, 'pvalues', pd.Series(dtype=float))
        robust_bse = getattr(robust_res, 'bse', pd.Series(dtype=float))
        robust_conf_int = robust_res.conf_int()
    except Exception:
        robust_params = pd.Series(dtype=float)
        robust_pvalues = pd.Series(dtype=float)
        robust_bse = pd.Series(dtype=float)
        robust_conf_int = None

    out = {}
    for term in terms:
        coef = safe_get(params, term)
        se = safe_get(bse, term)
        pval = safe_get(pvalues, term)

        # default 95% CI (Wald)
        if conf_int is not None and term in conf_int.index:
            ci_low, ci_high = conf_int.loc[term].iloc[0], conf_int.loc[term].iloc[1]
        else:
            ci_low = coef - 1.96 * se if not np.isnan(coef) and not np.isnan(se) else np.nan
            ci_high = coef + 1.96 * se if not np.isnan(coef) and not np.isnan(se) else np.nan

        # odds ratio and CI on OR scale
        or_point = np.exp(coef) if not np.isnan(coef) else np.nan
        or_ci = (np.exp(ci_low), np.exp(ci_high)) if not (np.isnan(ci_low) or np.isnan(ci_high)) else (np.nan, np.nan)

        # robust stats if present
        robust_coef = safe_get(robust_params, term)
        robust_se = safe_get(robust_bse, term)
        robust_p = safe_get(robust_pvalues, term)
        if robust_conf_int is not None and term in robust_conf_int.index:
            rci_low, rci_high = robust_conf_int.loc[term].iloc[0], robust_conf_int.loc[term].iloc[1]
        else:
            rci_low = robust_coef - 1.96 * robust_se if not np.isnan(robust_coef) and not np.isnan(robust_se) else np.nan
            rci_high = robust_coef + 1.96 * robust_se if not np.isnan(robust_coef) and not np.isnan(robust_se) else np.nan
        robust_or = np.exp(robust_coef) if not np.isnan(robust_coef) else np.nan
        robust_or_ci = (np.exp(rci_low), np.exp(rci_high)) if not (np.isnan(rci_low) or np.isnan(rci_high)) else (np.nan, np.nan)

        # significance decision: prefer robust p-value if available
        if not np.isnan(robust_p):
            significant = bool(robust_p < 0.05)
            used_p = robust_p
            used_se = robust_se
            used_ci = (rci_low, rci_high)
            used_or_ci = robust_or_ci
            used_est = robust_coef
        else:
            significant = bool(pval < 0.05) if not np.isnan(pval) else False
            used_p = pval
            used_se = se
            used_ci = (ci_low, ci_high)
            used_or_ci = or_ci
            used_est = coef

        out[term] = {
            'coef': float(coef) if not np.isnan(coef) else None,
            'se': float(se) if not np.isnan(se) else None,
            'pvalue': float(pval) if not np.isnan(pval) else None,
            'ci_95': (float(ci_low) if not np.isnan(ci_low) else None,
                      float(ci_high) if not np.isnan(ci_high) else None),
            'odds_ratio': float(or_point) if not np.isnan(or_point) else None,
            'odds_ratio_ci_95': (float(or_ci[0]) if not np.isnan(or_ci[0]) else None,
                                 float(or_ci[1]) if not np.isnan(or_ci[1]) else None),
            # robust alternatives
            'robust_coef': float(robust_coef) if not np.isnan(robust_coef) else None,
            'robust_se': float(robust_se) if not np.isnan(robust_se) else None,
            'robust_pvalue': float(robust_p) if not np.isnan(robust_p) else None,
            'robust_ci_95': (float(rci_low) if not np.isnan(rci_low) else None,
                             float(rci_high) if not np.isnan(rci_high) else None),
            'robust_odds_ratio': float(robust_or) if not np.isnan(robust_or) else None,
            'robust_odds_ratio_ci_95': (float(robust_or_ci[0]) if not np.isnan(robust_or_ci[0]) else None,
                                        float(robust_or_ci[1]) if not np.isnan(robust_or_ci[1]) else None),
            # decision
            'significant_at_0.05': significant,
            'used_pvalue_for_decision': float(used_p) if not np.isnan(used_p) else None,
            'used_estimate_for_decision': float(used_est) if not np.isnan(used_est) else None,
            'used_ci_for_decision': (float(used_ci[0]) if not np.isnan(used_ci[0]) else None,
                                     float(used_ci[1]) if not np.isnan(used_ci[1]) else None),
            'used_or_ci_for_decision': (float(used_or_ci[0]) if not np.isnan(used_or_ci[0]) else None,
                                        float(used_or_ci[1]) if not np.isnan(used_or_ci[1]) else None),
        }

    # Build a concise description
    # Interpret sign: coef >0 => higher log-odds for Female (Gender=1) vs Male (Gender=0)
    gender_info = out.get('Gender', {})
    gb_info = out.get('Gender_Black', {})

    if gender_info.get('used_estimate_for_decision') is None:
        desc = ("Could not find parameter estimates for 'Gender' in the supplied model output. "
                "Returned whatever numeric fields were available.")
    else:
        direction = 'higher' if gender_info['used_estimate_for_decision'] > 0 else 'lower' if gender_info['used_estimate_for_decision'] < 0 else 'no difference'
        sig_text = 'statistically significant' if gender_info['significant_at_0.05'] else 'not statistically significant'
        desc = (
            f"Gender (female=1 vs male=0): estimated log-odds effect = {gender_info['used_estimate_for_decision']:.4f}; "
            f"this corresponds to an odds ratio ≈ {np.exp(gender_info['used_estimate_for_decision']):.3f}. "
            f"The effect is {sig_text} at alpha=0.05 (p ≈ {gender_info['used_pvalue_for_decision']:.3g}). "
            f"In plain terms: females have {direction} odds of mortgage approval compared to males "
            f"after adjusting for controls. "
        )
        # Add interaction note if present
        if gb_info.get('used_estimate_for_decision') is not None:
            gb_dir = 'amplifies' if gb_info['used_estimate_for_decision'] > 0 else 'attenuates' if gb_info['used_estimate_for_decision'] < 0 else 'does not modify'
            gb_sig = 'statistically significant' if gb_info['significant_at_0.05'] else 'not statistically significant'
            desc += (
                f"The Gender × Black interaction estimate = {gb_info['used_estimate_for_decision']:.4f} "
                f"(OR ≈ {np.exp(gb_info['used_estimate_for_decision']):.3f}), which {gb_dir} the female effect for Black applicants; "
                f"this interaction is {gb_sig} (p ≈ {gb_info['used_pvalue_for_decision']:.3g})."
            )

    return {'object': out, 'description': desc}