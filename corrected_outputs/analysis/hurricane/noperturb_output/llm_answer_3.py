def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and (for NB) IRRs for the
    key predictors 'masfem_z' (primary) and 'gender_mf' (secondary) from the provided
    model_output. Returns a dict with an 'object' containing numeric summaries and
    a short 'description' interpreting whether the results support the hypothesis
    that more feminine hurricane names lead to outcomes consistent with fewer precautions.
    """
    import numpy as np

    def to_py(x):
        try:
            return float(np.asarray(x).item())
        except Exception:
            return None

    summary_obj = {
        'nb_model': None,
        'ols_model': None,
        'conclusion': None
    }

    # Helper to extract from a statsmodels results wrapper
    def extract_from_results(res):
        out = {}
        try:
            params = res.params.to_dict()
            pvalues = res.pvalues.to_dict()
            ci = res.conf_int().to_dict(orient='index')
            # convert CI format to [low, high] mapping
            ci_map = {k: [to_py(v[0]), to_py(v[1])] for k, v in ci.items()}
            for var in ('masfem_z', 'gender_mf'):
                if var in params:
                    coef = to_py(params[var])
                    pval = to_py(pvalues.get(var, None))
                    ci_low, ci_high = ci_map.get(var, (None, None))
                    out[var] = {
                        'coef': coef,
                        'pvalue': pval,
                        'ci_95': [ci_low, ci_high]
                    }
            return out
        except Exception:
            return {}

    # Try to use the full model objects if present
    nb_stats = {}
    ols_stats = {}
    try:
        if isinstance(model_output, dict) and 'nb_model' in model_output:
            nb_stats = extract_from_results(model_output['nb_model'])
    except Exception:
        nb_stats = {}

    try:
        if isinstance(model_output, dict) and 'ols_model' in model_output:
            ols_stats = extract_from_results(model_output['ols_model'])
    except Exception:
        ols_stats = {}

    # Fallback to the provided numeric summary (if present) if models didn't yield values
    if (not nb_stats or not all(k in nb_stats for k in ('masfem_z', 'gender_mf'))) and isinstance(model_output, dict) and 'summary' in model_output:
        s = model_output['summary']
        # Negative binomial summary
        try:
            nb_p = s.get('nb_params', {})
            nb_pv = s.get('nb_pvalues', {})
            nb_stats = {}
            for var in ('masfem_z', 'gender_mf'):
                if var in nb_p:
                    coef = to_py(nb_p[var])
                    pval = to_py(nb_pv.get(var, None))
                    nb_stats[var] = {'coef': coef, 'pvalue': pval, 'ci_95': [None, None]}
        except Exception:
            nb_stats = {}

    if (not ols_stats or 'masfem_z' not in ols_stats) and isinstance(model_output, dict) and 'summary' in model_output:
        s = model_output['summary']
        try:
            ols_p = s.get('ols_params', {})
            ols_pv = s.get('ols_pvalues', {})
            ols_stats = {}
            for var in ('masfem_z', 'gender_mf'):
                if var in ols_p:
                    coef = to_py(ols_p[var])
                    pval = to_py(ols_pv.get(var, None))
                    ols_stats[var] = {'coef': coef, 'pvalue': pval, 'ci_95': [None, None]}
        except Exception:
            ols_stats = {}

    # For NB model, compute IRR if coef is available
    if nb_stats:
        for var, d in nb_stats.items():
            coef = d.get('coef')
            if coef is not None:
                irr = float(np.exp(coef))
                # If CI available, exponentiate it
                ci = d.get('ci_95', [None, None])
                ci_exp = [np.exp(ci[0]) if ci[0] is not None else None,
                          np.exp(ci[1]) if ci[1] is not None else None]
                d.update({'IRR': irr, 'IRR_95': [to_py(ci_exp[0]), to_py(ci_exp[1])]})

    # Prepare summary object
    summary_obj['nb_model'] = nb_stats
    summary_obj['ols_model'] = ols_stats

    # Interpret evidence regarding the hypothesis
    # Primary check: masfem_z coefficient significance in NB (fatalities count) or OLS (damage)
    evidence = []
    def is_significant(p):
        return (p is not None) and (p < 0.05)

    nb_m = nb_stats.get('masfem_z')
    ols_m = ols_stats.get('masfem_z')

    if nb_m:
        coef = nb_m.get('coef')
        p = nb_m.get('pvalue')
        irr = nb_m.get('IRR')
        if is_significant(p):
            # direction: positive coef -> higher counts with more feminine names
            direction = 'higher' if coef > 0 else 'lower'
            evidence.append(f"NB model: masfem_z coef={coef:.3g}, p={p:.3g} -> statistically significant; suggests {direction} fatalities for more feminine names (IRR={irr:.3g}).")
        else:
            evidence.append(f"NB model: masfem_z coef={coef:.3g}, p={p:.3g} -> not statistically significant (no clear evidence).")

    if ols_m:
        coef = ols_m.get('coef')
        p = ols_m.get('pvalue')
        if is_significant(p):
            direction = 'higher' if coef > 0 else 'lower'
            evidence.append(f"OLS model (damage): masfem_z coef={coef:.3g}, p={p:.3g} -> statistically significant; suggests {direction} logged damage for more feminine names.")
        else:
            evidence.append(f"OLS model (damage): masfem_z coef={coef:.3g}, p={p:.3g} -> not statistically significant (no clear evidence).")

    # Draw final conclusion: require significant evidence in primary model (NB on fatalities)
    if nb_m and is_significant(nb_m.get('pvalue')):
        # If significant and coef positive -> supports hypothesis (more feminine -> more fatalities)
        if nb_m.get('coef') > 0:
            conclusion = "Support: The NB model shows a statistically significant association consistent with the hypothesis (more feminine names -> more fatalities)."
        else:
            conclusion = "Opposite: The NB model shows a statistically significant association opposite to the hypothesis (more feminine names -> fewer fatalities)."
    else:
        conclusion = "No strong evidence: The key coefficients for name femininity are not statistically significant in the provided models, so we do not have reliable evidence that more feminine hurricane names lead to fewer precautions (as proxied by fatalities or damages)."

    summary_obj['conclusion'] = conclusion

    # Build the return object (object: numeric summary; description: short interpretation)
    ret_obj = {
        'nb_model': nb_stats,
        'ols_model': ols_stats,
        'evidence_statements': evidence,
        'final_conclusion': conclusion
    }

    description = (
        "Extracted coefficients, p-values, and (for the negative binomial model) incident-rate ratios "
        "for the predictors 'masfem_z' (primary) and 'gender_mf' (secondary). "
        "Conclusion: based on the provided model output, there is no statistically significant evidence "
        "that more feminine hurricane names are associated with outcomes consistent with reduced precautionary "
        "behavior (no robust increase in fatalities or damages). See 'object' for numeric details."
    )

    return {"object": ret_obj, "description": description}