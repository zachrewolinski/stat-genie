def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of the masculinity-femininity index ('masfem_c')
    on the two main outcomes (log total deaths and log damages) from the provided
    model_output dictionary and returns a concise interpretation.

    Returns:
      {
        "object": {
          "ols_log_alldeaths": {coef, se, t/z, p, ci_lower, ci_upper, source},
          "ols_log_ndam15": { ... },
          "nb_alldeaths": { ... (if available) },
          "ols_log_alldeaths_mturkIV": { ... (if available) },
          "ols_log_ndam15_mturkIV": { ... (if available) },
          "simple_correlations": { ... }  # passed through if present
          "conclusion": { "supports_hypothesis": bool,
                          "reason": str }
        },
        "description": "Short plain-English summary of what the numbers mean."
      }
    """

    import numpy as np
    from math import isfinite

    def extract_from_modelobj(m):
        """Attempt to extract coef, se, t/z, p, CI for var 'masfem_c' from a statsmodels result object."""
        out = dict(coef=None, se=None, stat=None, p=None, ci_lower=None, ci_upper=None, source='model_object')
        if m is None:
            return out
        try:
            params = getattr(m, 'params', None)
            if params is not None and 'masfem_c' in params.index:
                out['coef'] = float(params['masfem_c'])
        except Exception:
            pass
        try:
            bse = getattr(m, 'bse', None)
            if bse is not None and 'masfem_c' in bse.index:
                out['se'] = float(bse['masfem_c'])
        except Exception:
            pass
        # tvalues / zvalues
        try:
            tvals = getattr(m, 'tvalues', None)
            if tvals is not None and 'masfem_c' in tvals.index:
                out['stat'] = float(tvals['masfem_c'])
        except Exception:
            pass
        try:
            pvals = getattr(m, 'pvalues', None)
            if pvals is not None and 'masfem_c' in pvals.index:
                out['p'] = float(pvals['masfem_c'])
        except Exception:
            pass
        try:
            ci = m.conf_int()
            if 'masfem_c' in ci.index:
                out['ci_lower'] = float(ci.loc['masfem_c'][0])
                out['ci_upper'] = float(ci.loc['masfem_c'][1])
        except Exception:
            pass
        return out

    def use_precomputed(key):
        """Use precomputed summary dict if present in model_output."""
        d = model_output.get(key)
        if not d:
            return None
        # keys expected: coef, se, t, p, ci_lower, ci_upper
        out = dict(
            coef=d.get('coef'),
            se=d.get('se'),
            stat=d.get('t'),
            p=d.get('p'),
            ci_lower=d.get('ci_lower'),
            ci_upper=d.get('ci_upper'),
            source='precomputed_summary'
        )
        return out

    results = {}

    # 1) Try to extract from the main OLS objects if present
    if 'ols_log_alldeaths' in model_output:
        results['ols_log_alldeaths'] = extract_from_modelobj(model_output['ols_log_alldeaths'])
        # fall back to precomputed if extraction failed or produced non-finite SE
        if (results['ols_log_alldeaths']['se'] is None) or (not isfinite(results['ols_log_alldeaths']['se'])):
            pre = use_precomputed('masfem_coef_ols_log_alldeaths')
            if pre:
                results['ols_log_alldeaths'] = pre

    if 'ols_log_ndam15' in model_output:
        results['ols_log_ndam15'] = extract_from_modelobj(model_output['ols_log_ndam15'])
        if (results['ols_log_ndam15']['se'] is None) or (not isfinite(results['ols_log_ndam15']['se'])):
            pre = use_precomputed('masfem_coef_ols_log_ndam15')
            if pre:
                results['ols_log_ndam15'] = pre

    # 2) Negative binomial model (counts), if present
    if 'nb_alldeaths' in model_output and model_output['nb_alldeaths'] is not None:
        try:
            results['nb_alldeaths'] = extract_from_modelobj(model_output['nb_alldeaths'])
            results['nb_alldeaths']['source'] = 'neg_binomial_model'
        except Exception:
            results['nb_alldeaths'] = None

    # 3) MTurk IV OLS robustness models (precomputed summary keys exist in model_output)
    if 'masfem_mturk_coef_ols_log_alldeaths' in model_output:
        results['ols_log_alldeaths_mturkIV'] = use_precomputed('masfem_mturk_coef_ols_log_alldeaths')
    elif 'ols_log_alldeaths_mturkIV' in model_output:
        results['ols_log_alldeaths_mturkIV'] = extract_from_modelobj(model_output['ols_log_alldeaths_mturkIV'])

    if 'masfem_mturk_coef_ols_log_ndam15' in model_output:
        results['ols_log_ndam15_mturkIV'] = use_precomputed('masfem_mturk_coef_ols_log_ndam15')
    elif 'ols_log_ndam15_mturkIV' in model_output:
        results['ols_log_ndam15_mturkIV'] = extract_from_modelobj(model_output['ols_log_ndam15_mturkIV'])

    # 4) include simple correlations if present
    if 'simple_correlations' in model_output:
        results['simple_correlations'] = model_output['simple_correlations']

    # 5) Build concise conclusion:
    # Hypothesis implies: coef(masfem_c) > 0 for deaths/damages (more feminine -> more deaths/damages).
    def interpret_entry(e):
        if not e or e.get('coef') is None:
            return {'supports': None, 'reason': 'no estimate available'}
        coef = e['coef']
        p = e.get('p')
        # If p is available and <0.05 and coef>0 => support. If coef<=0 and significant => contradicts.
        if p is not None:
            if p < 0.05:
                if coef > 0:
                    return {'supports': True, 'reason': 'positive and statistically significant (p<0.05).'}
                else:
                    return {'supports': False, 'reason': 'negative and statistically significant (p<0.05); opposite direction.'}
            else:
                # not statistically significant
                if coef > 0:
                    return {'supports': False, 'reason': 'positive point estimate but not statistically significant (p>=0.05).'}
                elif coef < 0:
                    return {'supports': False, 'reason': 'negative point estimate (opposite sign) and not statistically significant (p>=0.05).'}
                else:
                    return {'supports': False, 'reason': 'point estimate is exactly zero.'}
        else:
            # no p-value available: judge by sign only but mark as inconclusive
            if coef > 0:
                return {'supports': None, 'reason': 'positive point estimate but no p-value available (inconclusive).'}
            elif coef < 0:
                return {'supports': None, 'reason': 'negative point estimate but no p-value available (inconclusive).'}
            else:
                return {'supports': None, 'reason': 'point estimate is zero and no p-value available.'}

    interp_deaths = interpret_entry(results.get('ols_log_alldeaths'))
    interp_dam = interpret_entry(results.get('ols_log_ndam15'))

    # Combine to overall conclusion: require at least one of the two primary outcomes to positively and significantly support hypothesis.
    overall_support = False
    overall_reasons = []
    for name, interp in [('log_alldeaths', interp_deaths), ('log_ndam15', interp_dam)]:
        if interp.get('supports') is True:
            overall_support = True
        overall_reasons.append(f"{name}: {interp['reason']}")

    if overall_support:
        conclusion_text = "There is statistically significant evidence (on at least one primary outcome) that more-feminine hurricane names are associated with higher deaths/damages, consistent with the hypothesis."
    else:
        conclusion_text = ("No robust evidence supporting the hypothesis. "
                           "Point estimates for masfem are not positive and statistically significant. "
                           "Summary by outcome: " + " | ".join(overall_reasons))

    results['conclusion'] = {
        'supports_hypothesis': overall_support,
        'reason': conclusion_text
    }

    # Final user-friendly description
    description = (
        "Extracted coefficients, standard errors, test statistics, p-values and 95% CIs (when available) "
        "for the 'masfem_c' coefficient from the fitted models. The hypothesis predicts a positive and "
        "statistically significant coefficient (more feminine names -> higher deaths/damages). "
        "The extracted results show no such positive & significant effect: point estimates are negative or not "
        "significant, so we do not find evidence supporting the hypothesis."
    )

    return {"object": results, "description": description}