def extract_final_answer(model_output):
    """
    Extracts the gender effect on mortgage acceptance from the fitted model object returned
    by the modelling function. Returns a dictionary with keys:
      - "object": a dict with numeric results for (a) the female coefficient among non-Black
                  applicants and (b) the female effect among Black applicants (if the interaction
                  exists). Each entry contains coef, se, z, p, odds_ratio, and 95% CI for odds ratio.
                  Also includes a boolean 'significant' flag at alpha=0.05 and a short 'conclusion'.
      - "description": a plain-English summary of what the numbers mean.
    """
    import numpy as np
    try:
        from scipy import stats as _stats
        norm_cdf = _stats.norm.cdf
    except Exception:
        # fallback using scipy may not be available; use approximate normal cdf via math.erf
        import math as _math
        def norm_cdf(x):
            return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))

    # Helper to safely get attributes from wrapper or underlying result
    def _get_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Attempt to read parameters and covariance
    params = None
    cov = None
    bse = None
    pvalues = None

    # model_output is expected to expose .params, .cov_params() (callable), .bse, .pvalues
    if hasattr(model_output, "params"):
        params = model_output.params
    elif hasattr(model_output, "_res") and hasattr(model_output._res, "params"):
        params = model_output._res.params
    else:
        raise ValueError("Model output has no 'params' attribute.")

    # Covariance matrix (robust)
    if hasattr(model_output, "cov_params") and callable(model_output.cov_params):
        cov = model_output.cov_params()
    else:
        # try underlying result
        if hasattr(model_output, "_res") and hasattr(model_output._res, "cov_params"):
            cov = model_output._res.cov_params()
        else:
            raise ValueError("Model output has no 'cov_params' method to extract covariance matrix.")

    # bse and pvalues if available (for single-coefficient reporting)
    if hasattr(model_output, "bse"):
        bse = model_output.bse
    if hasattr(model_output, "pvalues"):
        pvalues = model_output.pvalues

    # Ensure params is a pandas Series or similar mapping
    try:
        param_index = list(params.index)
    except Exception:
        # treat params as dict-like
        param_index = list(params.keys())

    def _get_param(name):
        return params.get(name, None) if hasattr(params, "get") else (params[name] if name in param_index else None)

    def _get_cov_value(i, j):
        # cov may be a DataFrame, numpy array, or similar
        if cov is None:
            return None
        # If DataFrame-like with .loc
        try:
            # DataFrame or dict-of-dicts
            return cov.loc[i, j]
        except Exception:
            pass
        # If cov is numpy array, map indices via param_index
        try:
            ii = param_index.index(i)
            jj = param_index.index(j)
            return np.asarray(cov)[ii, jj]
        except Exception:
            raise ValueError(f"Cannot extract covariance for indices ({i}, {j}).")

    results = {}

    # Primary female effect (baseline group = non-Black if black is included as main effect)
    if 'female' not in param_index:
        raise ValueError("Model does not include a 'female' parameter; cannot evaluate gender effect.")

    female_coef = float(_get_param('female'))
    # bse for female either from bse attribute or via sqrt(cov(female,female))
    try:
        female_se = float(bse['female']) if (bse is not None and 'female' in getattr(bse, 'index', bse.keys() if hasattr(bse, 'keys') else [])) else float(np.sqrt(_get_cov_value('female', 'female')))
    except Exception:
        # last resort compute from cov
        female_se = float(np.sqrt(_get_cov_value('female', 'female')))

    female_z = female_coef / female_se if female_se != 0 else np.nan
    female_p = float(pvalues['female']) if (pvalues is not None and 'female' in getattr(pvalues, 'index', pvalues.keys() if hasattr(pvalues, 'keys') else [])) else 2 * (1 - norm_cdf(abs(female_z)))
    female_or = float(np.exp(female_coef))
    female_ci_low = float(np.exp(female_coef - 1.96 * female_se))
    female_ci_high = float(np.exp(female_coef + 1.96 * female_se))
    female_significant = (female_p < 0.05)

    results['female_nonblack'] = {
        'coef_log_odds': female_coef,
        'se': female_se,
        'z': female_z,
        'p_value': female_p,
        'odds_ratio': female_or,
        'odds_ratio_95CI': [female_ci_low, female_ci_high],
        'significant_at_0.05': bool(female_significant),
        'interpretation': (
            "Log-odds difference for female vs male among non-Black applicants "
            "(baseline where black=0). Positive coef means higher odds of acceptance for females; "
            "negative means lower odds."
        )
    }

    # If interaction exists, compute female effect among Black applicants as female + female_black
    if 'female_black' in param_index:
        fb_coef = float(_get_param('female_black'))
        # combined coef:
        female_black_coef = female_coef + fb_coef

        # variance = var(female) + var(female_black) + 2 * cov(female, female_black)
        try:
            var_f = _get_cov_value('female', 'female')
            var_fb = _get_cov_value('female_black', 'female_black')
            cov_f_fb = _get_cov_value('female', 'female_black')
            female_black_var = float(var_f + var_fb + 2.0 * cov_f_fb)
            female_black_se = float(np.sqrt(female_black_var))
        except Exception as e:
            # fallback: try symmetric positions if names differ in cov index, or raise
            raise ValueError("Could not compute variance for female + female_black. Error: " + str(e))

        female_black_z = female_black_coef / female_black_se if female_black_se != 0 else np.nan
        female_black_p = 2 * (1 - norm_cdf(abs(female_black_z)))
        female_black_or = float(np.exp(female_black_coef))
        female_black_ci_low = float(np.exp(female_black_coef - 1.96 * female_black_se))
        female_black_ci_high = float(np.exp(female_black_coef + 1.96 * female_black_se))
        female_black_significant = (female_black_p < 0.05)

        results['female_black'] = {
            'coef_log_odds': female_black_coef,
            'se': female_black_se,
            'z': female_black_z,
            'p_value': female_black_p,
            'odds_ratio': female_black_or,
            'odds_ratio_95CI': [female_black_ci_low, female_black_ci_high],
            'significant_at_0.05': bool(female_black_significant),
            'interpretation': (
                "Log-odds difference for female vs male among Black applicants (female + female_black). "
                "Positive coef means higher odds of acceptance for females; negative means lower odds."
            )
        }

        # Also provide the interaction term statistics (is the gender effect statistically different by race?)
        # Interaction significance can be read from params directly
        inter_coef = fb_coef
        try:
            inter_se = float(bse['female_black']) if (bse is not None and 'female_black' in getattr(bse, 'index', bse.keys() if hasattr(bse, 'keys') else [])) else float(np.sqrt(_get_cov_value('female_black', 'female_black')))
        except Exception:
            inter_se = float(np.sqrt(_get_cov_value('female_black', 'female_black')))
        inter_z = inter_coef / inter_se if inter_se != 0 else np.nan
        inter_p = float(pvalues['female_black']) if (pvalues is not None and 'female_black' in getattr(pvalues, 'index', pvalues.keys() if hasattr(pvalues, 'keys') else [])) else 2 * (1 - norm_cdf(abs(inter_z)))
        results['interaction_female_by_black'] = {
            'coef': inter_coef,
            'se': inter_se,
            'z': inter_z,
            'p_value': inter_p,
            'significant_at_0.05': bool(inter_p < 0.05),
            'interpretation': "If significant, indicates that the gender effect differs between Black and non-Black applicants."
        }

        # Short conclusion about gender effect heterogeneity
        if results['female_nonblack']['significant_at_0.05'] and results['female_black']['significant_at_0.05']:
            concl = "Female effect is statistically significant for both non-Black and Black applicants."
        elif results['female_nonblack']['significant_at_0.05'] and not results['female_black']['significant_at_0.05']:
            concl = "Female effect is statistically significant for non-Black applicants but not for Black applicants."
        elif not results['female_nonblack']['significant_at_0.05'] and results['female_black']['significant_at_0.05']:
            concl = "Female effect is statistically significant for Black applicants but not for non-Black applicants."
        else:
            concl = "No statistically significant female effect for either group at alpha=0.05."

    else:
        # No interaction term => same gender effect across race
        results['note'] = "No female_black interaction present; reported female effect applies across races in the model."
        # overall significance already in female_nonblack
        if results['female_nonblack']['significant_at_0.05']:
            concl = "Female applicants have statistically different odds of approval than male applicants (p < 0.05)."
        else:
            concl = "No statistically significant difference in odds of approval between female and male applicants (p >= 0.05)."

    # Build description string
    desc_lines = []
    desc_lines.append("Extracted estimates for the gender (female) effect on mortgage acceptance from the fitted logistic model.")
    desc_lines.append(
        "For non-Black applicants (baseline if 'black' variable is in the model): "
        f"coef(log-odds) = {results['female_nonblack']['coef_log_odds']:.4f}, "
        f"SE = {results['female_nonblack']['se']:.4f}, "
        f"z = {results['female_nonblack']['z']:.3f}, "
        f"p = {results['female_nonblack']['p_value']:.4f}, "
        f"OR = {results['female_nonblack']['odds_ratio']:.3f}, "
        f"OR 95% CI = [{results['female_nonblack']['odds_ratio_95CI'][0]:.3f}, {results['female_nonblack']['odds_ratio_95CI'][1]:.3f}]."
    )
    if 'female_black' in results:
        fb = results['female_black']
        desc_lines.append(
            "For Black applicants (female effect = female + female_black): "
            f"coef(log-odds) = {fb['coef_log_odds']:.4f}, SE = {fb['se']:.4f}, z = {fb['z']:.3f}, "
            f"p = {fb['p_value']:.4f}, OR = {fb['odds_ratio']:.3f}, "
            f"OR 95% CI = [{fb['odds_ratio_95CI'][0]:.3f}, {fb['odds_ratio_95CI'][1]:.3f}]."
        )
        inter = results['interaction_female_by_black']
        desc_lines.append(
            f"Interaction (female_black) coef = {inter['coef']:.4f}, p = {inter['p_value']:.4f} "
            "(tests whether the female effect differs by race)."
        )
    desc_lines.append("Conclusion: " + concl)

    description = " ".join(desc_lines)

    return {"object": results, "description": description}