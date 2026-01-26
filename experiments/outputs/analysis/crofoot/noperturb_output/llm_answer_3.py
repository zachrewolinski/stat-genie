def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels binary-logit results object
    (optionally with cluster-robust covariance already applied) and interprets
    the effects of:
      - RelSize_diff_z (and its interaction with FocalCloser)
      - dist_diff_z

    Returns:
      {
        "object": <dict of extracted numeric results and derived quantities>,
        "description": <text summary interpretation in the context of the task>
      }

    The function is defensive about parameter naming and will try a couple of
    plausible interaction-term name variants.
    """
    import numpy as np
    import pandas as pd
    try:
        from scipy.stats import norm
    except Exception:
        # approximate normal CDF if scipy is not available
        def _phi(x):
            return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0
        class _norm:
            @staticmethod
            def cdf(x): return _phi(x)
        norm = _norm()

    res = model_output

    # Helper to safely get attributes
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("Provided model_output does not have .params attribute (not a statsmodels results object).")
    # ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        params = pd.Series(params)

    # Attempt to get covariance matrix
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Try to obtain confidence intervals and standard errors robustly
    try:
        conf_int = res.conf_int()
        conf_int = pd.DataFrame(conf_int)  # ensure DataFrame
        # If number of rows matches params, align index
        if conf_int.shape[0] == len(params):
            conf_int.index = params.index
        # Standardize column names to ["2.5%", "97.5%"] if possible
        if conf_int.shape[1] >= 2:
            # Replace first two column names
            other_cols = list(conf_int.columns[2:]) if conf_int.shape[1] > 2 else []
            conf_int.columns = ["2.5%", "97.5%"] + other_cols
        else:
            # Unexpected shape; raise to go to fallback
            raise Exception("conf_int has unexpected shape")
    except Exception:
        # fallback: compute using normal approximation from bse if available
        bse_attr = getattr(res, "bse", None)
        if bse_attr is None:
            raise ValueError("Cannot obtain confidence intervals or standard errors from model_output.")
        # Ensure bse is a Series aligned with params
        if isinstance(bse_attr, pd.Series):
            bse_series = bse_attr.reindex(params.index)
        else:
            bse_series = pd.Series(bse_attr, index=params.index)
        ci_lower = params - 1.96 * bse_series
        ci_upper = params + 1.96 * bse_series
        conf_int = pd.DataFrame(np.vstack([ci_lower, ci_upper]).T, index=params.index, columns=["2.5%", "97.5%"])

    # pvalues
    pvalues = getattr(res, "pvalues", None)
    if pvalues is None:
        # cannot compute p-values without bse/cov
        bse_attr = getattr(res, "bse", None)
        if bse_attr is None:
            raise ValueError("Cannot obtain p-values or standard errors from model_output.")
        if isinstance(bse_attr, pd.Series):
            bse_series = bse_attr.reindex(params.index)
        else:
            bse_series = pd.Series(bse_attr, index=params.index)
        z = params / bse_series
        pvalues = 2 * (1 - norm.cdf(np.abs(z)))
        pvalues = pd.Series(pvalues, index=params.index)
    else:
        # ensure Series and aligned with params
        if not isinstance(pvalues, pd.Series):
            pvalues = pd.Series(pvalues, index=params.index)
        else:
            pvalues = pvalues.reindex(params.index)

    # Ensure bse_series exists for later use
    bse_attr = getattr(res, "bse", None)
    if bse_attr is None:
        bse_series = None
    else:
        if isinstance(bse_attr, pd.Series):
            bse_series = bse_attr.reindex(params.index)
        else:
            bse_series = pd.Series(bse_attr, index=params.index)

    # Identify coefficient names
    def find_name(possible_names):
        for n in possible_names:
            if n in params.index:
                return n
        return None

    name_size = find_name(['RelSize_diff_z'])
    name_dist = find_name(['dist_diff_z'])
    name_focalcloser = find_name(['FocalCloser'])
    # Interaction name variants
    name_inter = find_name(['RelSize_diff_z:FocalCloser', 'RelSize_diff_z: FocalCloser',
                            'RelSize_diff_z*FocalCloser', 'RelSize_diff_z:FocalCloser[T.1]',
                            'RelSize_diff_z:FocalCloser[T.True]'])

    if name_size is None or name_dist is None:
        raise ValueError("Could not find expected predictor names in model coefficients. Found: " + ", ".join(map(str, params.index)))

    # Basic extracted stats for main predictors and interaction (if present)
    def make_term_dict(term_name):
        if term_name is None or term_name not in params.index:
            return {
                'coef': None,
                'se': None,
                'pvalue': None,
                'ci_2.5%': None,
                'ci_97.5%': None,
                'odds_ratio': None,
                'or_ci_lower': None,
                'or_ci_upper': None,
            }
        coef = float(params.loc[term_name])
        se = None
        if bse_series is not None and term_name in bse_series.index:
            se = float(bse_series.loc[term_name])
        pval = float(pvalues.loc[term_name]) if term_name in pvalues.index else None
        ci_lower = None
        ci_upper = None
        if term_name in conf_int.index:
            # conf_int columns are standardized to "2.5%" and "97.5%"
            if "2.5%" in conf_int.columns and "97.5%" in conf_int.columns:
                ci_lower = float(conf_int.loc[term_name, "2.5%"])
                ci_upper = float(conf_int.loc[term_name, "97.5%"])
            else:
                # fallback to first two columns
                ci_lower = float(conf_int.iloc[conf_int.index.get_loc(term_name), 0])
                ci_upper = float(conf_int.iloc[conf_int.index.get_loc(term_name), 1])
        # use numpy.exp to avoid math.exp OverflowError on large values
        oratio = float(np.exp(coef)) if np.isfinite(coef) else float('inf')
        try:
            or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None and np.isfinite(ci_lower) else (float('inf') if ci_lower is not None and np.isposinf(ci_lower) else None)
        except Exception:
            or_ci_lower = None
        try:
            or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None and np.isfinite(ci_upper) else (float('inf') if ci_upper is not None and np.isposinf(ci_upper) else None)
        except Exception:
            or_ci_upper = None
        return {
            'coef': coef,
            'se': se,
            'pvalue': pval,
            'ci_2.5%': ci_lower,
            'ci_97.5%': ci_upper,
            'odds_ratio': oratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
        }

    results = {}
    results['RelSize_diff_z'] = make_term_dict(name_size)
    results['dist_diff_z'] = make_term_dict(name_dist)
    if name_focalcloser is not None:
        results['FocalCloser'] = make_term_dict(name_focalcloser)
    if name_inter is not None:
        results['RelSize_x_FocalCloser'] = make_term_dict(name_inter)
    else:
        results['RelSize_x_FocalCloser'] = None

    # Compute marginal effect of RelSize_diff_z when FocalCloser = 0 and =1
    coef_size = float(params.loc[name_size])
    coef_inter = float(params.loc[name_inter]) if name_inter is not None and name_inter in params.index else 0.0

    marg0_coef = coef_size
    marg1_coef = coef_size + coef_inter

    # Compute standard errors for marginals using covariance matrix if available
    def lincomb_var(names, coefs):
        """
        Var(sum_i c_i * beta_i) using cov matrix.
        names: list of param names
        coefs: list of multipliers (1 or so)
        """
        if cov is None:
            return None
        cov_df = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov, index=params.index, columns=params.index)
        var = 0.0
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                # if any name missing from cov_df, cannot compute
                if ni not in cov_df.index or nj not in cov_df.columns:
                    return None
                var += coefs[i] * coefs[j] * float(cov_df.loc[ni, nj])
        return var

    var_marg0 = lincomb_var([name_size], [1.0]) if name_size in params.index else None
    var_marg1 = None
    if name_inter is not None and name_inter in params.index:
        var_marg1 = lincomb_var([name_size, name_inter], [1.0, 1.0])
    elif name_inter is None:
        # no interaction term; marg1 same as marg0
        var_marg1 = var_marg0

    def make_marginal_entry(coef, var):
        if var is None:
            se = None
            z = None
            p = None
            ci_low = None
            ci_high = None
            or_ci_low = None
            or_ci_high = None
        else:
            se = np.sqrt(var)
            # protect division by zero
            if se == 0:
                z = None
                p = None
                ci_low = coef
                ci_high = coef
            else:
                z = coef / se
                p = 2 * (1 - norm.cdf(abs(z)))
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se
            try:
                or_ci_low = float(np.exp(ci_low)) if ci_low is not None and np.isfinite(ci_low) else (float('inf') if ci_low is not None and np.isposinf(ci_low) else None)
            except Exception:
                or_ci_low = None
            try:
                or_ci_high = float(np.exp(ci_high)) if ci_high is not None and np.isfinite(ci_high) else (float('inf') if ci_high is not None and np.isposinf(ci_high) else None)
            except Exception:
                or_ci_high = None
        try:
            odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else float('inf')
        except Exception:
            odds_ratio = None
        return {
            'logit_coef': float(coef),
            'se': float(se) if se is not None else None,
            'z': float(z) if z is not None else None,
            'pvalue': float(p) if p is not None else None,
            'ci_2.5%': float(ci_low) if ci_low is not None else None,
            'ci_97.5%': float(ci_high) if ci_high is not None else None,
            'odds_ratio': odds_ratio,
            'or_ci_lower': float(or_ci_low) if or_ci_low is not None else None,
            'or_ci_upper': float(or_ci_high) if or_ci_high is not None else None,
        }

    marg0 = make_marginal_entry(marg0_coef, var_marg0)
    marg1 = make_marginal_entry(marg1_coef, var_marg1)

    results['marginal_effects_of_RelSize'] = {
        'when_FocalCloser_0': marg0,
        'when_FocalCloser_1': marg1,
        'interaction_term_name': name_inter
    }

    # Simple decision statements about significance (alpha=0.05) for primary terms
    def significance(p):
        if p is None:
            return 'unknown'
        return 'significant' if p < 0.05 else 'not_significant'

    size_p = results['RelSize_diff_z']['pvalue']
    dist_p = results['dist_diff_z']['pvalue']
    inter_p = results['RelSize_x_FocalCloser']['pvalue'] if results['RelSize_x_FocalCloser'] is not None else None

    # Build a concise text description (guard formatting if values missing)
    lines = []
    # Rel size main effect (when FocalCloser=0)
    lines.append("Relative group size (RelSize_diff_z):")
    coef_v = results['RelSize_diff_z']['coef']
    or_v = results['RelSize_diff_z']['odds_ratio']
    or_low = results['RelSize_diff_z']['or_ci_lower']
    or_high = results['RelSize_diff_z']['or_ci_upper']
    lines.append(" - Coefficient (log-odds) = {}, OR = {}, 95% CI OR = [{}, {}].".format(
        f"{coef_v:.4f}" if coef_v is not None else "NA",
        f"{or_v:.3f}" if or_v is not None else "NA",
        f"{or_low:.3f}" if or_low is not None else "NA",
        f"{or_high:.3f}" if or_high is not None else "NA",
    ))
    lines.append(" - p-value = {} -> {} (this is the effect when FocalCloser=0).".format(
        f"{size_p:.3g}" if size_p is not None else "NA",
        significance(size_p)
    ))

    # Interaction interpretation
    if name_inter is not None:
        inter_coef = results['RelSize_x_FocalCloser']['coef']
        lines.append("")
        lines.append("Interaction (RelSize_diff_z x FocalCloser):")
        lines.append(" - Interaction coef (log-odds) = {}, p = {} -> {}.".format(
            f"{inter_coef:.4f}" if inter_coef is not None else "NA",
            f"{inter_p:.3g}" if inter_p is not None else "NA",
            significance(inter_p)
        ))
        lines.append(" - Marginal effect of RelSize when FocalCloser=0: logit = {}, OR = {}.".format(
            f"{marg0['logit_coef']:.4f}" if marg0.get('logit_coef') is not None else "NA",
            f"{marg0['odds_ratio']:.3f}" if marg0.get('odds_ratio') is not None else "NA"
        ))
        if marg0.get('pvalue') is not None:
            lines.append("   p = {}".format(f"{marg0['pvalue']:.3g}"))
        lines.append(" - Marginal effect of RelSize when FocalCloser=1: logit = {}, OR = {}.".format(
            f"{marg1['logit_coef']:.4f}" if marg1.get('logit_coef') is not None else "NA",
            f"{marg1['odds_ratio']:.3f}" if marg1.get('odds_ratio') is not None else "NA"
        ))
        if marg1.get('pvalue') is not None:
            lines.append("   p = {}".format(f"{marg1['pvalue']:.3g}"))
        if inter_p is not None and inter_p < 0.05:
            lines.append(" -> Interpretation: The effect of being relatively larger on the odds of winning depends on whether the focal group is closer to its home-range center.")
        else:
            lines.append(" -> Interpretation: No strong evidence that the size advantage differs by whether the focal group is closer to home (interaction not significant).")
    else:
        lines.append("")
        lines.append("No interaction term was found in the fitted model, so the reported RelSize_diff_z effect applies regardless of FocalCloser status.")

    # Dist (location) effect
    lines.append("")
    coef_v = results['dist_diff_z']['coef']
    or_v = results['dist_diff_z']['odds_ratio']
    or_low = results['dist_diff_z']['or_ci_lower']
    or_high = results['dist_diff_z']['or_ci_upper']
    lines.append("Relative contest location (dist_diff_z):")
    lines.append(" - Coefficient (log-odds) = {}, OR = {}, 95% CI OR = [{}, {}].".format(
        f"{coef_v:.4f}" if coef_v is not None else "NA",
        f"{or_v:.3f}" if or_v is not None else "NA",
        f"{or_low:.3f}" if or_low is not None else "NA",
        f"{or_high:.3f}" if or_high is not None else "NA",
    ))
    lines.append(" - p-value = {} -> {}.".format(
        f"{dist_p:.3g}" if dist_p is not None else "NA",
        significance(dist_p)
    ))
    lines.append(" -> Interpretation: Positive dist_diff_z means the focal group is closer to its home center relative to the other; a OR > 1 indicates greater odds of winning when focal is closer.")

    description = "\n".join(lines)

    return {
        "object": results,
        "description": description
    }