def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for the
    predictors of interest from a fitted statsmodels MixedLMResultsWrapper,
    and computes simple slopes for Age when Help = 0 and Help = 1 (if possible).
    Returns a dictionary with keys "object" and "description".
    """
    import numpy as np
    from math import isnan
    try:
        from scipy import stats as _scipystats
        _norm_cdf = _scipystats.norm.cdf
    except Exception:
        # fallback to approximate normal cdf using statsmodels if scipy not present
        try:
            from statsmodels.distributions.empirical_distribution import ECDF
            # very unlikely fallback; but define a simple normal cdf approximation
            import math
            def _norm_cdf(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        except Exception:
            def _norm_cdf(x):
                # very rough fallback
                return 0.5

    # Terms we care about
    focal_terms = ['Age_c', 'Sex_M', 'Help_Y', 'Age_c:Help_Y']

    out = {'terms': {}, 'age_simple_effects': {}}
    try:
        params = model_output.params  # pandas Series
    except Exception as e:
        return {
            "object": None,
            "description": f"Error: could not read model params from model_output: {e}"
        }

    # Get bse, pvalues, conf_int, cov_params if available
    bse = getattr(model_output, 'bse', None)
    pvalues = getattr(model_output, 'pvalues', None)
    try:
        conf_int_df = model_output.conf_int()
    except Exception:
        conf_int_df = None
    try:
        covp = model_output.cov_params()
    except Exception:
        covp = None

    # Helper to get values safely
    def get_from_series(series, name):
        if series is None:
            return np.nan
        try:
            return series.get(name, np.nan)
        except Exception:
            try:
                return series.loc[name]
            except Exception:
                return np.nan

    # Build per-term results
    for term in focal_terms:
        coef = get_from_series(params, term)
        se = get_from_series(bse, term)
        pval = get_from_series(pvalues, term)

        # If pval is missing but coef and se available, approximate using normal
        if (pval is None or (isinstance(pval, float) and np.isnan(pval))) and (not (isinstance(coef, float) and np.isnan(coef))) and (not (isinstance(se, float) and np.isnan(se))) and se != 0:
            try:
                z = coef / se
                pval = 2 * (1 - _norm_cdf(abs(z)))
            except Exception:
                pval = np.nan

        # Confidence interval
        ci_lower = ci_upper = np.nan
        if conf_int_df is not None:
            try:
                # conf_int_df may have columns [0,1] and index of term names
                row = conf_int_df.loc[term]
                ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
            except Exception:
                ci_lower = ci_upper = np.nan
        else:
            # approximate using normal 1.96
            if (not (isinstance(coef, float) and np.isnan(coef))) and (not (isinstance(se, float) and np.isnan(se))):
                try:
                    ci_lower = coef - 1.96 * se
                    ci_upper = coef + 1.96 * se
                except Exception:
                    ci_lower = ci_upper = np.nan

        out['terms'][term] = {
            'coef': None if (isinstance(coef, float) and np.isnan(coef)) else float(coef),
            'se': None if (isinstance(se, float) and np.isnan(se)) else (float(se) if se is not None else None),
            'pvalue': None if (isinstance(pval, float) and np.isnan(pval)) else (float(pval) if pval is not None else None),
            '95% CI': [None if (isinstance(ci_lower, float) and np.isnan(ci_lower)) else ci_lower,
                       None if (isinstance(ci_upper, float) and np.isnan(ci_upper)) else ci_upper],
            'significant_p_lt_0.05': False if (pval is None or (isinstance(pval, float) and np.isnan(pval))) else (float(pval) < 0.05)
        }

    # Compute simple slopes for Age when Help_Y = 0 and Help_Y = 1:
    # Age_effect_help0 = coef(Age_c)
    # Age_effect_help1 = coef(Age_c) + coef(Age_c:Help_Y)
    age_coef = out['terms']['Age_c']['coef']
    inter_coef = out['terms']['Age_c:Help_Y']['coef']
    age_se = out['terms']['Age_c']['se']
    inter_se = out['terms']['Age_c:Help_Y']['se']

    # default values
    out['age_simple_effects']['Help=0'] = {
        'slope': None, 'se': None, 'pvalue': None, '95% CI': [None, None], 'significant_p_lt_0.05': None
    }
    out['age_simple_effects']['Help=1'] = {
        'slope': None, 'se': None, 'pvalue': None, '95% CI': [None, None], 'significant_p_lt_0.05': None
    }

    # Fill Help=0 (just Age_c)
    if age_coef is not None:
        out['age_simple_effects']['Help=0']['slope'] = age_coef
        out['age_simple_effects']['Help=0']['se'] = age_se
        if age_se is not None:
            try:
                z = age_coef / age_se
                p = 2 * (1 - _norm_cdf(abs(z)))
                out['age_simple_effects']['Help=0']['pvalue'] = float(p)
                out['age_simple_effects']['Help=0']['significant_p_lt_0.05'] = (p < 0.05)
                out['age_simple_effects']['Help=0']['95% CI'] = [age_coef - 1.96 * age_se, age_coef + 1.96 * age_se]
            except Exception:
                pass

    # Fill Help=1 (Age_c + interaction)
    if (age_coef is not None) and (inter_coef is not None):
        slope1 = age_coef + inter_coef
        out['age_simple_effects']['Help=1']['slope'] = slope1

        # Compute standard error of sum using covariance matrix if available
        se_sum = None
        if covp is not None:
            try:
                var_age = float(covp.loc['Age_c', 'Age_c'])
                var_inter = float(covp.loc['Age_c:Help_Y', 'Age_c:Help_Y'])
                cov_ai = float(covp.loc['Age_c', 'Age_c:Help_Y'])
                se_sum = np.sqrt(var_age + var_inter + 2 * cov_ai)
            except Exception:
                se_sum = None

        # Fallback: approximate by sqrt(se_age^2 + se_inter^2) (assumes zero cov)
        if se_sum is None:
            if (age_se is not None) and (inter_se is not None):
                try:
                    se_sum = np.sqrt(age_se ** 2 + inter_se ** 2)
                except Exception:
                    se_sum = None

        if se_sum is not None:
            out['age_simple_effects']['Help=1']['se'] = float(se_sum)
            try:
                z1 = slope1 / se_sum
                p1 = 2 * (1 - _norm_cdf(abs(z1)))
                out['age_simple_effects']['Help=1']['pvalue'] = float(p1)
                out['age_simple_effects']['Help=1']['significant_p_lt_0.05'] = (p1 < 0.05)
                out['age_simple_effects']['Help=1']['95% CI'] = [slope1 - 1.96 * se_sum, slope1 + 1.96 * se_sum]
            except Exception:
                pass
        else:
            # Could not compute se; leave pvalue and CI as None
            out['age_simple_effects']['Help=1']['se'] = None

    # Build a short textual description that highlights significance and interpretation
    significant_terms = [t for t, v in out['terms'].items() if v['significant_p_lt_0.05']]
    desc_lines = []
    if len(significant_terms) == 0:
        desc_lines.append("None of Age_c, Sex_M, Help_Y, or their Age_c:Help_Y interaction showed p < 0.05 according to the model output.")
    else:
        desc_lines.append("Predictors with p < 0.05: " + ", ".join(significant_terms) + ".")

    # Add brief interpretation for each focal term (if available)
    for term in focal_terms:
        info = out['terms'][term]
        if info['coef'] is None:
            desc_lines.append(f"{term}: no estimate available in the model output.")
            continue
        sig_text = "statistically significant" if info['significant_p_lt_0.05'] else "not statistically significant"
        desc_lines.append(
            f"{term}: coef={info['coef']:.4g}, se={info['se']:.4g} (95% CI [{None if info['95% CI'][0] is None else format(info['95% CI'][0], '.4g')}, {None if info['95% CI'][1] is None else format(info['95% CI'][1], '.4g')}]), p={None if info['pvalue'] is None else format(info['pvalue'], '.4g')}; {sig_text}."
        )

    # Add interpretation of simple age slopes
    for key in ['Help=0', 'Help=1']:
        s = out['age_simple_effects'][key]
        if s['slope'] is None:
            desc_lines.append(f"Age effect when {key}: could not be computed from model output.")
        else:
            sig_text = "significant" if (s.get('significant_p_lt_0.05') is True) else "not significant"
            desc_lines.append(
                f"Age effect when {key}: slope={s['slope']:.4g}, se={None if s.get('se') is None else format(s.get('se'), '.4g')}, p={None if s.get('pvalue') is None else format(s.get('pvalue'), '.4g')}; {sig_text}."
            )

    description = " ".join(desc_lines)

    return {
        "object": out,
        "description": description
    }