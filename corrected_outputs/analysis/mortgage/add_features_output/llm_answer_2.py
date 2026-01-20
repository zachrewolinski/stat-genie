def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' treatment effect from a fitted statsmodels Logit result.

    Returns:
        dict with keys:
          - "object": dict of extracted numeric statistics (coef, se, p-value, CI, odds ratio, OR CI,
                      marginal effect and its CI if available)
          - "description": short textual interpretation in the context of the mortgage approval task
    """
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    if 'female' not in res.params.index:
        raise ValueError("The fitted model does not contain a parameter named 'female'.")

    # Extract coefficient, SE, p-value, and coefficient CI
    coef = float(res.params['female'])
    se = float(res.bse['female']) if ('female' in getattr(res, "bse", {}).index) else None
    pval = float(res.pvalues['female']) if ('female' in getattr(res, "pvalues", {}).index) else None

    try:
        ci_series = res.conf_int().loc['female']
        ci_lower = float(ci_series[0])
        ci_upper = float(ci_series[1])
    except Exception:
        ci_lower = None
        ci_upper = None

    # Odds ratio and CI
    try:
        odds_ratio = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if (ci_lower is not None) else None
        or_ci_upper = float(np.exp(ci_upper)) if (ci_upper is not None) else None
    except Exception:
        odds_ratio = or_ci_lower = or_ci_upper = None

    # Average marginal effect (change in probability) if available
    me = None
    me_se = None
    me_p = None
    me_ci_lower = None
    me_ci_upper = None
    try:
        margeff = res.get_margeff(at='overall')  # average marginal effects
        sf = margeff.summary_frame()
        # typical column names: 'dy/dx', 'Std. Err.', 'z', 'P>|z|', '[0.025', '0.975]'
        if 'female' in sf.index:
            row = sf.loc['female']
            me = float(row.get('dy/dx', row.get('dydx', row.get('Delta', np.nan))))
            me_se = float(row.get('Std. Err.', row.get('std err', np.nan)))
            me_p = float(row.get('P>|z|', row.get('p', np.nan)))
            # confidence interval column names can vary slightly
            # try common names:
            me_ci_lower = float(row.get('[0.025', row.get('0.025', np.nan)))
            me_ci_upper = float(row.get('0.975]', row.get('0.975', np.nan)))
        else:
            # fallback: try to index by position (if only one effect)
            if sf.shape[0] == 1:
                row = sf.iloc[0]
                me = float(row.iloc[0])
                me_se = float(row.iloc[1]) if sf.shape[1] > 1 else None
    except Exception:
        # If get_margeff is not available or fails, leave marginal effects as None
        pass

    result_object = {
        "coef_log_odds": coef,
        "std_err": se,
        "p_value": pval,
        "coef_ci_lower": ci_lower,
        "coef_ci_upper": ci_upper,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_lower": or_ci_lower,
        "odds_ratio_ci_upper": or_ci_upper,
        "avg_marginal_effect": me,
        "avg_marginal_effect_se": me_se,
        "avg_marginal_effect_p_value": me_p,
        "avg_marginal_effect_ci_lower": me_ci_lower,
        "avg_marginal_effect_ci_upper": me_ci_upper,
    }

    # Short interpretation
    signif_text = ""
    try:
        if pval is not None:
            if pval < 0.001:
                signif_text = "statistically significant (p < 0.001)"
            elif pval < 0.01:
                signif_text = "statistically significant (p < 0.01)"
            elif pval < 0.05:
                signif_text = "statistically significant (p < 0.05)"
            else:
                signif_text = "not statistically significant (p >= 0.05)"
    except Exception:
        signif_text = "statistical significance could not be determined"

    # Directional interpretation for odds
    direction = "increased" if (odds_ratio is not None and odds_ratio > 1) else "decreased" if (odds_ratio is not None and odds_ratio < 1) else "no clear direction"

    description_parts = [
        f"The logistic regression coefficient for 'female' (log-odds) = {coef:.4f}" if coef is not None else "Coefficient unavailable",
        f"(SE = {se:.4f})" if se is not None else "",
        f", p = {pval:.4g}." if pval is not None else ".",
        f"Interpretation: being female is associated with {direction} odds of mortgage approval; odds ratio = {odds_ratio:.3f}" if odds_ratio is not None else "",
    ]
    if or_ci_lower is not None and or_ci_upper is not None:
        description_parts.append(f"(95% CI for OR: [{or_ci_lower:.3f}, {or_ci_upper:.3f}]).")

    if me is not None:
        description_parts.append(
            f"Average marginal effect (change in predicted probability) = {me:.4f} (SE = {me_se:.4f}),"
            + (f" 95% CI = [{me_ci_lower:.4f}, {me_ci_upper:.4f}]." if (me_ci_lower is not None and me_ci_upper is not None) else "")
        )

    description_parts.append(f"Statistical significance: {signif_text}.")

    description = " ".join([p for p in description_parts if p])

    return {"object": result_object, "description": description}