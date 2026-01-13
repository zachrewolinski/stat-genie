def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of age on choosing the majority option
    from a fitted logistic regression result object (or the placeholder EmptyModelResult).
    Returns a dict with keys:
      - "object": a nested dict with numeric results (coefficients, SEs, p-values, CIs),
      - "description": short interpretation about whether age effects (linear/quadratic)
                       or site-by-age interactions are present / interpretable.
    The function is defensive and works if model_output.params contains NaNs or when
    there are no observations (EmptyModelResult).
    """
    import numpy as np
    import pandas as pd

    # Prepare output container
    extracted = {
        'Age_c': None,
        'Age_c_sq': None,
        'Site_age_slopes': None,
        'nobs': None,
        'aic': None,
        'available_params': []
    }

    # Check that model_output exposes the expected attributes
    if not hasattr(model_output, 'params'):
        return {
            'object': extracted,
            'description': "The provided model_output has no 'params' attribute; cannot extract statistics."
        }

    params = model_output.params
    pvalues = getattr(model_output, 'pvalues', None)
    bse = getattr(model_output, 'bse', None)
    nobs = getattr(model_output, 'nobs', None)
    aic = getattr(model_output, 'aic', None)

    # Record nobs and aic if available
    try:
        extracted['nobs'] = int(nobs) if nobs is not None and not (isinstance(nobs, float) and np.isnan(nobs)) else None
    except Exception:
        extracted['nobs'] = None
    try:
        extracted['aic'] = float(aic) if aic is not None and not (isinstance(aic, float) and np.isnan(aic)) else None
    except Exception:
        extracted['aic'] = None

    # Helper to safely extract stats for a parameter name
    def stat_for(name):
        if name is None or name == '':
            return None
        if name in params.index:
            coef = params.loc[name]
            pv = pvalues.loc[name] if (pvalues is not None and name in pvalues.index) else None
            se = bse.loc[name] if (bse is not None and name in bse.index) else None

            # Convert to float if possible, otherwise None
            def safe_float(x):
                try:
                    if pd.isnull(x):
                        return None
                    return float(x)
                except Exception:
                    return None

            coef_f = safe_float(coef)
            se_f = safe_float(se)
            pv_f = safe_float(pv)

            ci = None
            if (coef_f is not None) and (se_f is not None):
                ci = (coef_f - 1.96 * se_f, coef_f + 1.96 * se_f)

            extracted['available_params'].append(name)

            return {
                'coef': coef_f,
                'se': se_f,
                'pvalue': pv_f,
                '95_CI': (float(ci[0]), float(ci[1])) if ci is not None else None
            }
        else:
            return None

    # Extract main age effects
    extracted['Age_c'] = stat_for('Age_c')
    extracted['Age_c_sq'] = stat_for('Age_c_sq')

    # Identify site dummies and interactions in the parameter names
    param_names = list(params.index)
    site_dummies = [n for n in param_names if n.startswith('Site_') and not n.endswith('_x_Age')]
    interactions = [n for n in param_names if n.endswith('_x_Age')]

    # Build site-specific slopes: reference site's slope = Age_c; other sites = Age_c + interaction_coef
    site_slopes = {}
    # Reference site (the omitted baseline) slope is simply Age_c
    site_slopes['reference_site'] = {
        'slope_estimate': extracted['Age_c']['coef'] if extracted['Age_c'] is not None else None,
        'slope_se': extracted['Age_c']['se'] if extracted['Age_c'] is not None else None,
        'slope_pvalue': extracted['Age_c']['pvalue'] if extracted['Age_c'] is not None else None,
        'notes': 'This is the slope for the reference (omitted) site in the dummy coding.'
    }

    # For each interaction present, compute site-specific slope as sum of Age_c and interaction coefficient.
    # Note: exact SE/p-value for the sum requires covariance; we report the point estimate and component stats.
    for inter in interactions:
        site_var = inter.replace('_x_Age', '')
        inter_stat = stat_for(inter)
        age_coef = extracted['Age_c']['coef'] if extracted['Age_c'] is not None else None

        if age_coef is not None and inter_stat is not None and inter_stat.get('coef') is not None:
            slope = age_coef + inter_stat['coef']
            site_slopes[site_var] = {
                'slope_estimate': slope,
                'slope_se': None,  # cannot compute without covariance matrix
                'slope_pvalue_interaction_term': inter_stat.get('pvalue'),
                'components': {
                    'Age_c': extracted['Age_c'],
                    inter: inter_stat
                },
                'notes': ("Slope = Age_c + interaction_coef. SE for the sum not computed here "
                          "because covariance between Age_c and interaction term is not available.")
            }
        else:
            site_slopes[site_var] = {
                'slope_estimate': None,
                'slope_se': None,
                'slope_pvalue_interaction_term': inter_stat.get('pvalue') if inter_stat is not None else None,
                'components': {
                    'Age_c': extracted['Age_c'],
                    inter: inter_stat
                },
                'notes': 'Incomplete components to compute slope estimate (missing Age_c and/or interaction coef).'
            }

    extracted['Site_age_slopes'] = site_slopes

    # Interpretation rules (conservative):
    significant_findings = []
    alpha = 0.05
    if extracted['Age_c'] and extracted['Age_c'].get('pvalue') is not None and extracted['Age_c']['pvalue'] < alpha:
        significant_findings.append('linear age effect (Age_c)')

    if extracted['Age_c_sq'] and extracted['Age_c_sq'].get('pvalue') is not None and extracted['Age_c_sq']['pvalue'] < alpha:
        significant_findings.append('quadratic age effect (Age_c_sq)')

    sig_interactions = []
    for inter in interactions:
        pv = None
        if pvalues is not None and inter in pvalues.index:
            try:
                pv = float(pvalues.loc[inter]) if not pd.isnull(pvalues.loc[inter]) else None
            except Exception:
                pv = None
        if pv is not None and pv < alpha:
            sig_interactions.append(inter)
    if sig_interactions:
        significant_findings.append('site-by-age interactions: ' + ', '.join(sig_interactions))

    # Compose description
    if extracted['nobs'] == 0:
        description = ("No observations were used to fit the model; all parameter estimates are unavailable "
                       "or NaN. Cannot draw conclusions about developmental change or cultural variation.")
    elif (extracted['Age_c'] is None and extracted['Age_c_sq'] is None and not interactions):
        description = ("Model does not contain Age_c, Age_c_sq, or site-by-age interaction parameters "
                       "in its results; nothing to extract about age-related development.")
    elif not significant_findings:
        description = ("No statistically significant evidence (alpha=0.05) that reliance on the majority changes "
                       "with age (neither linear nor quadratic), nor that it varies by site (no significant interactions). "
                       "See 'object' for the estimated coefficients, standard errors, and p-values.")
    else:
        description = ("Statistically significant effects detected at alpha=0.05: " +
                       "; ".join(significant_findings) +
                       ". Consult the 'object' field for effect sizes, component statistics, and site-specific slopes. "
                       "Note: SEs and p-values for sums (e.g., Age_c + interaction) are not computed here because the "
                       "covariance matrix is required for exact inference.")

    return {
        'object': extracted,
        'description': description
    }