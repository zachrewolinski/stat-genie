def extract_final_answer(model_output):
    """
    Extracts statistics related to the effect of the 'Female' indicator from a fitted model output.
    Returns a dictionary with keys:
      - "object": dict of numeric results (coef, se, z, p, odds_ratio, odds_ratio_ci)
      - "description": brief plain-language interpretation of what the numbers mean
    
    The function is defensive: it will try multiple places for parameters, p-values and CIs
    and will compute approximations when necessary.
    """
    import numpy as np
    import math

    target = 'Female'
    res = model_output.get('result') if isinstance(model_output, dict) else None
    odds_provided = model_output.get('odds_ratio') if isinstance(model_output, dict) else None
    conf_odds_provided = model_output.get('conf_odds') if isinstance(model_output, dict) else None

    coef = None
    se = None
    z = None
    p = None
    ci_log_lower = None
    ci_log_upper = None
    odds_ratio = None
    odds_ci_lower = None
    odds_ci_upper = None

    # Try to extract from statsmodels result-like object if available
    if res is not None:
        # params
        try:
            params = getattr(res, 'params', None)
            if params is not None and target in params.index:
                coef = float(params[target])
        except Exception:
            coef = None

        # standard error
        try:
            bse = getattr(res, 'bse', None)
            if bse is not None and target in bse.index:
                se = float(bse[target])
        except Exception:
            se = None

        # p-value (direct)
        try:
            pvals = getattr(res, 'pvalues', None)
            if pvals is not None and target in pvals.index:
                p = float(pvals[target])
        except Exception:
            p = None

        # confidence interval on log-odds scale
        try:
            conf = res.conf_int()
            # conf_int returns a DataFrame; columns may be [0,1] or named
            if target in conf.index:
                # handle both possible column namings
                try:
                    ci_log_lower = float(conf.loc[target].iloc[0])
                    ci_log_upper = float(conf.loc[target].iloc[1])
                except Exception:
                    # try named columns
                    try:
                        ci_log_lower = float(conf.loc[target, '2.5%'])
                        ci_log_upper = float(conf.loc[target, '97.5%'])
                    except Exception:
                        ci_log_lower = None
                        ci_log_upper = None
        except Exception:
            ci_log_lower = None
            ci_log_upper = None

    # If p-value missing but coef and se available, compute Wald z and approximate p using normal
    if p is None and coef is not None and se is not None and se != 0:
        try:
            from scipy import stats as _st
            z = coef / se
            p = float(2 * (1.0 - _st.norm.cdf(abs(z))))
        except Exception:
            # fallback: use standard normal CDF approx via math.erf if scipy not available
            z = coef / se
            try:
                # normal cdf via erf
                cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
                p = float(2 * (1 - cdf))
            except Exception:
                p = None

    # If z hasn't been set but we have coef and se, compute it anyway
    if z is None and coef is not None and se is not None and se != 0:
        z = coef / se

    # Odds ratio: prefer explicit odds_ratio Series if available, else exp(coef)
    if odds_provided is not None:
        try:
            # odds_provided may be a pandas Series or dict-like
            if hasattr(odds_provided, 'get'):
                # dict-like: try .get first
                or_val = odds_provided.get(target, None)
            else:
                # if Series-like with indexing
                or_val = odds_provided[target] if target in odds_provided.index else None
            if or_val is not None and not (isinstance(or_val, float) and (math.isnan(or_val))):
                odds_ratio = float(or_val)
        except Exception:
            odds_ratio = None

    if odds_ratio is None and coef is not None:
        try:
            odds_ratio = float(np.exp(coef))
        except Exception:
            odds_ratio = None

    # Odds ratio CI: prefer provided conf_odds, else exponentiate log-odds CI if available
    if conf_odds_provided is not None:
        try:
            if hasattr(conf_odds_provided, 'loc') and target in conf_odds_provided.index:
                # try named columns first
                try:
                    lower = conf_odds_provided.loc[target, '2.5%']
                    upper = conf_odds_provided.loc[target, '97.5%']
                    odds_ci_lower = float(lower) if not (lower is None or (isinstance(lower, float) and math.isnan(lower))) else None
                    odds_ci_upper = float(upper) if not (upper is None or (isinstance(upper, float) and math.isnan(upper))) else None
                except Exception:
                    # fallback to positional
                    try:
                        row = conf_odds_provided.loc[target]
                        odds_ci_lower = float(row.iloc[0])
                        odds_ci_upper = float(row.iloc[1])
                    except Exception:
                        odds_ci_lower = None
                        odds_ci_upper = None
        except Exception:
            odds_ci_lower = None
            odds_ci_upper = None

    if (odds_ci_lower is None or odds_ci_upper is None) and ci_log_lower is not None and ci_log_upper is not None:
        try:
            odds_ci_lower = float(np.exp(ci_log_lower))
            odds_ci_upper = float(np.exp(ci_log_upper))
        except Exception:
            odds_ci_lower = None
            odds_ci_upper = None

    # Build the returned numeric object
    numeric_object = {
        'coef_log_odds': None if coef is None else float(coef),
        'std_error': None if se is None else float(se),
        'z_stat': None if z is None else float(z),
        'p_value': None if p is None else float(p),
        'odds_ratio': None if odds_ratio is None else float(odds_ratio),
        'odds_ratio_ci_lower': None if odds_ci_lower is None else float(odds_ci_lower),
        'odds_ratio_ci_upper': None if odds_ci_upper is None else float(odds_ci_upper)
    }

    # Plain-language interpretation
    # We avoid causal language and report this as an association from the fitted logistic model.
    lines = []
    lines.append("Association of applicant being female with odds of mortgage acceptance (multivariable logistic regression).")
    if numeric_object['odds_ratio'] is not None:
        lines.append(f"- Estimated odds ratio for Female (female vs male), controlling for covariates: {numeric_object['odds_ratio']:.4g}.")
        if numeric_object['odds_ratio'] < 1:
            lines.append("  Interpretation: female applicants have lower estimated odds of acceptance compared to male applicants (holding covariates constant).")
        elif numeric_object['odds_ratio'] > 1:
            lines.append("  Interpretation: female applicants have higher estimated odds of acceptance compared to male applicants (holding covariates constant).")
        else:
            lines.append("  Interpretation: no estimated difference in odds.")

        if numeric_object['odds_ratio_ci_lower'] is not None and numeric_object['odds_ratio_ci_upper'] is not None:
            lines.append(f"  95% CI for odds ratio: [{numeric_object['odds_ratio_ci_lower']:.4g}, {numeric_object['odds_ratio_ci_upper']:.4g}].")
        else:
            lines.append("  95% CI for odds ratio not available from the fitted object.")

        if numeric_object['p_value'] is not None:
            if numeric_object['p_value'] < 0.05:
                lines.append(f"  The effect is statistically significant at the 5% level (p = {numeric_object['p_value']:.3g}).")
            else:
                lines.append(f"  The effect is not statistically significant at the 5% level (p = {numeric_object['p_value']:.3g}).")
        else:
            lines.append("  p-value not available from the fitted object.")
    else:
        lines.append("Could not extract an odds ratio for 'Female' from the model output. Check that the fitted object contains parameter estimates.")

    description = " ".join(lines)

    return {
        "object": numeric_object,
        "description": description
    }