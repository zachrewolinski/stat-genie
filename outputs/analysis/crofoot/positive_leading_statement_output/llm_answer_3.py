def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM (or a robustified wrapper)
    predicting 'win' from RelSizeDiff_z, RelDist_z and their interaction.

    Returns a dictionary with:
      - "object": a dict containing:
          * coef_table: DataFrame with coefficients, robust SEs, z, p, 95% CI, odds ratios and OR CIs
          * marginal_effects: dict with marginal effects (log-odds coef, SE, z, p, 95% CI, OR, OR CI)
            for:
              - effect of RelSizeDiff_z at RelDist_z = [-1, 0, +1] (i.e., -1SD, mean, +1SD)
              - effect of RelDist_z at RelSizeDiff_z = [-1, 0, +1]
      - "description": short text describing what the returned numbers mean and how to interpret them.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    res = model_output  # expected: statsmodels results wrapper (possibly with clustered cov)
    # Ensure necessary attributes exist
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")

    params = res.params
    # Robust/clustered standard errors will be present in .bse and pvalues if get_robustcov_results was used.
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    # conf_int method or attribute
    try:
        ci = res.conf_int()
        # conf_int returns array-like with two columns
        ci_df = pd.DataFrame(ci, index=params.index, columns=['ci_lower', 'ci_upper'])
    except Exception:
        # fallback: compute from coef +/- 1.96*bse if bse exists
        if bse is None:
            raise ValueError("Cannot obtain confidence intervals or standard errors from model_output.")
        ci_df = pd.DataFrame({
            'ci_lower': params - 1.96 * bse,
            'ci_upper': params + 1.96 * bse
        }, index=params.index)

    # Build coefficient table for focal terms
    terms_of_interest = ['RelSizeDiff_z', 'RelDist_z', 'RelSizeDiff_z:RelDist_z']
    available_terms = [t for t in terms_of_interest if t in params.index]

    coef_rows = []
    for t in available_terms:
        coef = params[t]
        se = float(bse[t]) if (bse is not None and t in bse.index) else np.nan
        if (not np.isnan(se)) and np.isfinite(se) and se != 0.0:
            z = coef / se
        else:
            z = np.nan
        p = float(pvalues[t]) if (pvalues is not None and t in pvalues.index) else np.nan
        ci_low = float(ci_df.loc[t, 'ci_lower'])
        ci_high = float(ci_df.loc[t, 'ci_upper'])
        or_est = np.exp(coef)
        or_ci_low = np.exp(ci_low)
        or_ci_high = np.exp(ci_high)
        coef_rows.append({
            'term': t,
            'coef': float(coef),
            'se': se,
            'z': float(z) if not np.isnan(z) else np.nan,
            'p': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'OR': or_est,
            'OR_ci_lower': or_ci_low,
            'OR_ci_upper': or_ci_high
        })
    coef_table = pd.DataFrame(coef_rows).set_index('term')

    # For interaction interpretation: compute marginal effect of one predictor conditional on values of the other
    # Use delta method: var(beta1 + val * beta_int) = Var(beta1) + val^2 Var(beta_int) + 2*val Cov(beta1,beta_int)
    # Need covariance matrix
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    def marginal_effect(primary, moderator, moderator_values):
        """
        primary: name of primary coefficient (e.g., 'RelSizeDiff_z')
        moderator: name of moderator coefficient (e.g., 'RelDist_z')
        moderator_values: iterable of numeric values at which to evaluate the marginal effect
        Returns a DataFrame with rows for each moderator value
        """
        rows = []
        inter_name = f"{primary}:{moderator}"
        # statsmodels uses ':' ordering exactly as in formula, our interaction term name is 'RelSizeDiff_z:RelDist_z'
        if inter_name not in params.index:
            # Try reversed order (just in case)
            inter_name_rev = f"{moderator}:{primary}"
            if inter_name_rev in params.index:
                inter_name = inter_name_rev
            else:
                # No interaction term available
                raise KeyError(f"Interaction term not found for {primary} and {moderator}.")
        for val in moderator_values:
            coef_primary = params[primary]
            coef_inter = params[inter_name]
            # marginal log-odds effect
            me = coef_primary + coef_inter * val
            # variance via delta method
            if cov is not None and primary in cov.index and inter_name in cov.index:
                var = cov.loc[primary, primary] + (val ** 2) * cov.loc[inter_name, inter_name] + 2 * val * cov.loc[primary, inter_name]
                se_me = np.sqrt(var) if (var >= 0 and np.isfinite(var)) else np.nan
            else:
                se_me = np.nan
            if (not np.isnan(se_me)) and np.isfinite(se_me) and se_me != 0.0:
                z_me = me / se_me
                p_me = 2 * (1 - norm.cdf(abs(z_me)))
                ci_low = me - 1.96 * se_me
                ci_high = me + 1.96 * se_me
            else:
                z_me = np.nan
                p_me = np.nan
                ci_low = np.nan
                ci_high = np.nan
            or_me = np.exp(me) if not np.isnan(me) else np.nan
            or_ci_low = np.exp(ci_low) if not np.isnan(ci_low) else np.nan
            or_ci_high = np.exp(ci_high) if not np.isnan(ci_high) else np.nan
            rows.append({
                f'{moderator}': val,
                'marginal_logodds': float(me),
                'se': float(se_me) if not np.isnan(se_me) else np.nan,
                'z': float(z_me) if not np.isnan(z_me) else np.nan,
                'p': float(p_me) if not np.isnan(p_me) else np.nan,
                'ci_lower': float(ci_low) if not np.isnan(ci_low) else np.nan,
                'ci_upper': float(ci_high) if not np.isnan(ci_high) else np.nan,
                'OR': float(or_me) if not np.isnan(or_me) else np.nan,
                'OR_ci_lower': float(or_ci_low) if not np.isnan(or_ci_low) else np.nan,
                'OR_ci_upper': float(or_ci_high) if not np.isnan(or_ci_high) else np.nan
            })
        return pd.DataFrame(rows).set_index(moderator)

    marginal_effects = {}
    # Evaluate at -1, 0, +1 (approx -1SD, mean, +1SD since variables were standardized)
    eval_points = [-1.0, 0.0, 1.0]
    try:
        if ('RelSizeDiff_z' in params.index) and ('RelDist_z' in params.index) and (('RelSizeDiff_z:RelDist_z' in params.index) or ('RelDist_z:RelSizeDiff_z' in params.index)):
            # Effect of RelSizeDiff_z at values of RelDist_z
            marginal_effects['RelSize_effect_at_RelDist'] = marginal_effect('RelSizeDiff_z', 'RelDist_z', eval_points)
            # Effect of RelDist_z at values of RelSizeDiff_z
            marginal_effects['RelDist_effect_at_RelSize'] = marginal_effect('RelDist_z', 'RelSizeDiff_z', eval_points)
    except KeyError:
        # If interaction missing, skip marginal effects
        marginal_effects['note'] = 'Interaction term not found; marginal effects not computed.'

    results = {
        'coef_table': coef_table,
        'marginal_effects': marginal_effects,
        # include a small covariance subset for the three terms if available
        'cov_subset': (cov.loc[available_terms, available_terms] if (cov is not None and set(available_terms).issubset(cov.index)) else None)
    }

    description = (
        "Returned object contains: (1) coef_table: coefficient estimates for RelSizeDiff_z, RelDist_z, "
        "and their interaction (if present) with robust SEs, z-stats, p-values, 95% CIs, and odds ratios; "
        "(2) marginal_effects: estimated marginal effect (log-odds change) of RelSizeDiff_z evaluated at "
        "RelDist_z = -1, 0, +1 and vice versa, with SEs and CIs computed by the delta method (requires covariance matrix). "
        "Interpretation: a positive coefficient means higher values of that predictor increase the log-odds (and odds) "
        "that the focal group wins. When an interaction is present, main-effect coefficients are conditional on the moderator being zero (the mean); "
        "use the marginal_effects table to see how effects change across values of the other predictor. "
        "Check p-values and 95% CIs to assess statistical evidence (p < 0.05 and CI not including 0 for log-odds, or not including 1 for OR, indicate statistical significance)."
    )

    return {'object': results, 'description': description}