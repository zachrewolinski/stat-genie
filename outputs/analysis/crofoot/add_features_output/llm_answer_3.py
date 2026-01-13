def extract_final_answer(model_output):
    """
    Extracts coefficients, cluster-robust standard errors, p-values, confidence intervals,
    odds ratios, and marginal effects for the key predictors from a fitted statsmodels
    logit model object (the object returned by the `model` function in the prompt).

    Returns a dictionary with keys:
      - "object": nested dict of extracted numeric results (see structure below)
      - "description": brief explanation of the meaning of the numbers

    The "object" dict contains entries for:
      - terms: main coefficients for z_SizeDiff, z_RelDist, and their interaction
               (if present) with coef, cluster_se, z, p, 95% CI, OR, OR 95% CI
      - marginal_size_at_relDist: marginal effect of z_SizeDiff at z_RelDist = -1, 0, +1
               (coef, se, z, p, 95% CI, OR, OR 95% CI)
    """
    import numpy as np
    import pandas as pd
    from math import exp
    from scipy import stats

    res = model_output

    # Try to obtain parameter table with cluster-robust SEs if attached
    summary_df = getattr(res, 'clustered_summary', None)

    if summary_df is None:
        # Build summary_df from available info
        params = getattr(res, 'params', None)
        # prefer cluster robust bse if attached
        bse = getattr(res, 'clustered_bse', None)
        if bse is None:
            # fallback to default bse (not cluster-robust)
            bse = getattr(res, 'bse', None)

        # try to obtain covariance matrix if available (clustered or default)
        cov = getattr(res, 'cluster_cov', None)
        if cov is None:
            try:
                cov = res.cov_params()
            except Exception:
                cov = None

        if params is None or bse is None:
            raise ValueError("Cannot find parameters or standard errors on the model_output object.")

        # construct DataFrame
        z_vals = params / bse
        p_vals = 2 * stats.norm.sf(np.abs(z_vals))
        conf_low = params - 1.96 * bse
        conf_high = params + 1.96 * bse
        summary_df = pd.DataFrame({
            'coef': params,
            'cluster_se': bse,
            'z': z_vals,
            'P>|z|': p_vals,
            '2.5%': conf_low,
            '97.5%': conf_high
        })
    else:
        # ensure it's a DataFrame
        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)

        # also try to get covariance matrix for linear combinations
        cov = getattr(res, 'cluster_cov', None)
        if cov is None:
            try:
                cov = res.cov_params()
            except Exception:
                cov = None

    # helper to find parameter names (accounts for possible naming differences)
    params_index = list(summary_df.index.astype(str))
    def _find_term(*candidates):
        for cand in candidates:
            if cand in params_index:
                return cand
        return None

    term_size = _find_term('z_SizeDiff', 'z_SizeDiff')
    term_rel = _find_term('z_RelDist', 'z_RelDist')
    term_inter = _find_term('z_SizeDiff:z_RelDist', 'z_RelDist:z_SizeDiff', 'z_SizeDiff*z_RelDist')

    # build results for present terms
    terms_out = {}
    for name, term in [('z_SizeDiff', term_size), ('z_RelDist', term_rel), ('z_SizeDiff:z_RelDist', term_inter)]:
        if term is not None:
            row = summary_df.loc[term]
            coef = float(row['coef'])
            se = float(row['cluster_se'])
            z = float(row['z'])
            p = float(row['P>|z|'])
            ci_low = float(row['2.5%'])
            ci_high = float(row['97.5%'])
            or_ = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            terms_out[name] = {
                'term_name': term,
                'coef': coef,
                'cluster_se': se,
                'z': z,
                'p': p,
                '95%_CI_coef': (ci_low, ci_high),
                'odds_ratio': or_,
                '95%_CI_OR': (or_ci_low, or_ci_high)
            }
        else:
            terms_out[name] = None

    # Compute marginal effect of z_SizeDiff at z_RelDist = -1, 0, +1 (standardized units)
    marginal_effects = {}
    if term_size is not None:
        beta_size = float(summary_df.loc[term_size, 'coef'])
        # if interaction exists, use it; else interaction coef = 0
        if term_inter is not None:
            beta_inter = float(summary_df.loc[term_inter, 'coef'])
        else:
            beta_inter = 0.0

        # need covariance entries for variance of linear combination
        # cov must be a DataFrame or array with parameter order matching summary_df.index
        cov_df = None
        if cov is not None:
            try:
                # if cov is a numpy array, convert to DataFrame with same index
                if isinstance(cov, np.ndarray):
                    cov_df = pd.DataFrame(cov, index=summary_df.index, columns=summary_df.index)
                else:
                    # assume DataFrame-like
                    cov_df = pd.DataFrame(cov)
                    # ensure indices align
                    if not all(str(i) in summary_df.index.astype(str) for i in cov_df.index):
                        # attempt to reindex
                        cov_df = cov_df.reindex(index=summary_df.index, columns=summary_df.index)
            except Exception:
                cov_df = None

        for val in [-1.0, 0.0, 1.0]:
            eff_coef = beta_size + beta_inter * val
            # compute se for linear combination: Var(beta_size + val*beta_inter)
            if cov_df is not None and (term_size in cov_df.index) and (term_inter in cov_df.index if term_inter is not None else True):
                try:
                    var_size = float(cov_df.loc[term_size, term_size])
                    if term_inter is not None:
                        var_inter = float(cov_df.loc[term_inter, term_inter])
                        cov_si = float(cov_df.loc[term_size, term_inter])
                    else:
                        var_inter = 0.0
                        cov_si = 0.0
                    eff_var = var_size + (val ** 2) * var_inter + 2.0 * val * cov_si
                    eff_se = float(np.sqrt(max(eff_var, 0.0)))
                except Exception:
                    eff_se = None
            else:
                # fallback: approximate using cluster_ses ignoring covariance
                se_size = float(summary_df.loc[term_size, 'cluster_se'])
                se_inter = float(summary_df.loc[term_inter, 'cluster_se']) if (term_inter is not None) else 0.0
                eff_se = float(np.sqrt(se_size**2 + (val**2) * se_inter**2))

            if eff_se is not None and eff_se > 0:
                eff_z = eff_coef / eff_se
                eff_p = float(2 * stats.norm.sf(abs(eff_z)))
                ci_low = eff_coef - 1.96 * eff_se
                ci_high = eff_coef + 1.96 * eff_se
            else:
                eff_z = None
                eff_p = None
                ci_low = None
                ci_high = None

            marginal_effects[val] = {
                'z_RelDist_value': val,
                'marginal_coef_for_z_SizeDiff': eff_coef,
                'se': eff_se,
                'z': eff_z,
                'p': eff_p,
                '95%_CI_coef': (ci_low, ci_high),
                'odds_ratio': (np.exp(eff_coef) if eff_coef is not None else None),
                '95%_CI_OR': (np.exp(ci_low) if ci_low is not None else None,
                              np.exp(ci_high) if ci_high is not None else None)
            }
    else:
        marginal_effects = None

    # Compose output object
    output_object = {
        'terms': terms_out,
        'marginal_size_at_relDist': marginal_effects,
        'notes': (
            "Positive coefficient means higher log-odds of the focal group winning. "
            "Odds ratio > 1 means higher odds of focal win. The interaction term (if present) "
            "indicates whether the effect of relative group size depends on contest location. "
            "Marginal effects report the effective coefficient of z_SizeDiff when z_RelDist = -1, 0, +1."
        )
    }

    # Human-readable description (keeps interpretation general because actual numeric
    # significance depends on the extracted p-values).
    descr_lines = [
        "I extracted coefficient estimates, cluster-robust standard errors (if available), z-values, two-sided p-values,",
        "95% confidence intervals on coefficients, and odds ratios for the key predictors:",
        "  - z_SizeDiff (relative group size)",
        "  - z_RelDist (relative contest location)",
        "  - z_SizeDiff:z_RelDist (interaction)",
        "",
        "Additionally, I computed the marginal effect of relative group size at z_RelDist = -1, 0, +1 (standardized units),",
        "including approximate standard errors, p-values, confidence intervals, and odds ratios.",
        "",
        "How to interpret the numbers you'll find in 'object':",
        "  - coef > 0 => increases log-odds of the focal group winning; coef < 0 => decreases.",
        "  - odds_ratio = exp(coef): value >1 increases odds of focal win; <1 decreases odds.",
        "  - If the interaction term is statistically significant (small p-value), the effect of group size depends on location.",
        "  - The marginal effects show how the size effect changes when the focal group is relatively farther from (-1),",
        "    average (0), or closer to (+1) its home-range center compared to the opponent.",
    ]
    description = "\n".join(descr_lines)

    return {"object": output_object, "description": description}