def extract_final_answer(model_output):
    """
    Extract key statistics for the 'female' coefficient from a fitted statsmodels Logit result.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coef, se, p-value, CI, odds ratio, OR CI,
                  and if available the average marginal effect (AME) with its SE and p-value).
      - "description": A short, plain-language interpretation of those statistics.

    Expects model_output to be a statsmodels BinaryResultsWrapper (Logit results).
    """
    import numpy as np

    res = model_output

    # Ensure 'female' is in the result
    if 'female' not in getattr(res, "params").index:
        raise KeyError("The fitted model does not contain a parameter named 'female'.")

    # Coefficient, SE, p-value
    coef = float(res.params['female'])
    se = float(res.bse['female'])
    pval = float(res.pvalues['female'])

    # 95% CI on the log-odds scale
    try:
        ci = res.conf_int().loc['female']
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # Fallback if conf_int returns an array-like without .loc
        conf = res.conf_int()
        try:
            idx = list(res.params.index).index('female')
            ci_lower = float(conf[idx, 0])
            ci_upper = float(conf[idx, 1])
        except Exception:
            ci_lower = float('nan')
            ci_upper = float('nan')

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else float('nan')
    or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else float('nan')

    # Try to compute average marginal effect for 'female' (probability change)
    ame = None
    ame_se = None
    ame_p = None
    try:
        # get_margeff may raise for some result objects; handle gracefully
        me = res.get_margeff(at='overall', method='dydx')
        me_df = me.summary_frame()
        # summary_frame index should include 'female'
        if 'female' in me_df.index:
            ame = float(me_df.loc['female', 'dy/dx'])
            # Column names can vary slightly; attempt common alternatives
            if 'Std. Err.' in me_df.columns:
                ame_se = float(me_df.loc['female', 'Std. Err.'])
            elif 'std err' in [c.lower() for c in me_df.columns]:
                # find the matching column case-insensitively
                for c in me_df.columns:
                    if c.lower() == 'std err':
                        ame_se = float(me_df.loc['female', c])
                        break
            if 'P>|z|' in me_df.columns:
                ame_p = float(me_df.loc['female', 'P>|z|'])
            elif 'p' in [c.lower() for c in me_df.columns]:
                for c in me_df.columns:
                    if c.lower() == 'p':
                        ame_p = float(me_df.loc['female', c])
                        break
    except Exception:
        # If marginal effects cannot be computed, leave them as None
        pass

    # Helper to safely format numeric or None values for the description
    def _fmt(val, fmt):
        if val is None:
            return "NA"
        try:
            # treat NaN as NA
            if isinstance(val, float) and np.isnan(val):
                return "NA"
            return format(val, fmt)
        except Exception:
            try:
                return str(val)
            except Exception:
                return "NA"

    # Build the numeric object to return
    result_object = {
        'coef_log_odds': coef,
        'std_err': se,
        'p_value': pval,
        'ci_95_log_odds': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'ci_95_odds_ratio': [or_ci_lower, or_ci_upper],
        'average_marginal_effect': ame,        # change in probability (if available)
        'ame_std_err': ame_se,
        'ame_p_value': ame_p
    }

    # Build a concise interpretation
    signif_text = "statistically significant" if (pval is not None and not np.isnan(pval) and pval < 0.05) else "not statistically significant"
    desc_lines = [
        f"The model coefficient for 'female' is {coef:.4f} (SE = {se:.4f}, p = {_fmt(pval, '.4g')}),",
        f"which is {signif_text} at the 5% level.",
        f"This coefficient is on the log-odds scale: exp(coef) = odds ratio = {_fmt(odds_ratio, '.3f')}",
        f"with a 95% CI for the odds ratio of [{_fmt(or_ci_lower, '.3f')}, {_fmt(or_ci_upper, '.3f')}].",
        "Interpretation: holding the included controls constant, the odds ratio above compares",
        "the odds of loan acceptance for female applicants to male applicants (female vs male)."
    ]
    if ame is not None or ame_se is not None or ame_p is not None:
        desc_lines.append(
            "Average marginal effect (female) = "
            f"{_fmt(ame, '.4f')} (SE = {_fmt(ame_se, '.4f')}, p = {_fmt(ame_p, '.4g')}), "
            "i.e., the estimated change in probability of approval associated with being female."
        )

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}