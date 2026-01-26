def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, test statistics, p-values, 95% CIs,
    and an interpretable percent-change for the three predictors of interest
    (age, sex_m, help_yes) from the provided model_output dict.

    model_output is expected to contain at least:
      - 'mixedlm': a statsmodels MixedLMResults or wrapper
      - 'ols_cluster': a statsmodels RegressionResults (possibly robust-cov results)

    Returns:
      {
        "object": {
           "mixedlm": { predictor: {coef, se, stat, pvalue, ci_lower, ci_upper,
                                     percent_change, significant}, ...},
           "ols_cluster": { ... },
           "predictors": [list of predictors examined]
        },
        "description": "Short interpretation of the numbers and what they mean"
      }
    """
    import numpy as np
    from math import isfinite
    try:
        from scipy import stats as _scipy_stats
        _norm_sf = _scipy_stats.norm.sf
    except Exception:
        # fallback: approximate using numpy (less precise), but should rarely be needed
        def _norm_sf(x):
            # survival function for std normal using erf
            import math
            return 0.5 * (1.0 - math.erf(x / np.sqrt(2.0)))

    predictors = ['age', 'sex_m', 'help_yes']
    results = {'mixedlm': {}, 'ols_cluster': {}}

    def safe_items_from_mixed(mixed):
        """
        Try to extract fixed-effect params, standard errors, and p-values
        from a MixedLMResults object using several possible attribute names.
        """
        # Try common attributes
        fe_params = None
        bse = None
        pvalues = None
        # preferred attributes in recent statsmodels:
        if hasattr(mixed, 'fe_params'):
            fe_params = getattr(mixed, 'fe_params')
        elif hasattr(mixed, 'params'):
            # params can include both fixed and random; but often for MixedLM it's fine
            fe_params = getattr(mixed, 'params')

        # standard errors for fixed effects
        if hasattr(mixed, 'bse_fe'):
            bse = getattr(mixed, 'bse_fe')
        elif hasattr(mixed, 'bse'):
            bse = getattr(mixed, 'bse')

        # p-values may or may not be present
        if hasattr(mixed, 'pvalues'):
            pvalues = getattr(mixed, 'pvalues')

        return fe_params, bse, pvalues

    def process_model(model, model_type):
        """
        Given a model result object and its type ('mixed' or 'ols'),
        extract stats for predictors.
        """
        out = {}
        if model is None:
            return out

        if model_type == 'mixed':
            fe_params, bse, pvals = safe_items_from_mixed(model)
            # convert to dict-like if it's a pandas Series
            try:
                keys = list(fe_params.index)
            except Exception:
                keys = None

            for pred in predictors:
                entry = {}
                coef = None
                se = None
                pval = None
                stat = None
                ci_lo = None
                ci_hi = None

                # Get coef
                try:
                    if hasattr(fe_params, 'get') and (pred in fe_params):
                        coef = float(fe_params[pred])
                    else:
                        # try attribute access or numeric indexing
                        coef = float(fe_params.loc[pred])
                except Exception:
                    # try params as fallback
                    try:
                        coef = float(model.params.get(pred, np.nan))
                    except Exception:
                        coef = np.nan

                # standard error
                try:
                    if hasattr(bse, 'get') and (pred in bse):
                        se = float(bse[pred])
                    else:
                        se = float(bse.loc[pred])
                except Exception:
                    # fallback: try to get from model.bse if available
                    try:
                        se = float(model.bse.get(pred))
                    except Exception:
                        se = np.nan

                # p-value
                try:
                    if pvals is not None:
                        if hasattr(pvals, 'get') and (pred in pvals):
                            pval = float(pvals[pred])
                        else:
                            pval = float(pvals.loc[pred])
                    else:
                        # compute z-stat and p from normal approximation
                        if (se is not None) and isfinite(se) and (se != 0) and isfinite(coef):
                            stat = coef / se
                            pval = float(2.0 * _norm_sf(abs(stat)))
                except Exception:
                    pval = np.nan

                # stat if not computed
                try:
                    if stat is None and (se is not None) and isfinite(se) and (se != 0) and isfinite(coef):
                        stat = coef / se
                except Exception:
                    stat = None

                # 95% CI using normal approx
                try:
                    if (se is not None) and isfinite(se) and isfinite(coef):
                        ci_lo = coef - 1.96 * se
                        ci_hi = coef + 1.96 * se
                except Exception:
                    ci_lo = ci_hi = np.nan

                # package
                entry['coef'] = None if coef is None or not isfinite(coef) else float(coef)
                entry['se'] = None if se is None or not isfinite(se) else float(se)
                entry['stat'] = None if stat is None or not isfinite(stat) else float(stat)
                entry['pvalue'] = None if pval is None or not isfinite(pval) else float(pval)
                entry['ci_lower'] = None if ci_lo is None or not isfinite(ci_lo) else float(ci_lo)
                entry['ci_upper'] = None if ci_hi is None or not isfinite(ci_hi) else float(ci_hi)
                # interpret as percent change because DV is log(nuts/sec)
                try:
                    if entry['coef'] is not None:
                        entry['percent_change'] = float((np.exp(entry['coef']) - 1.0) * 100.0)
                    else:
                        entry['percent_change'] = None
                except Exception:
                    entry['percent_change'] = None

                entry['significant'] = (entry['pvalue'] is not None) and (entry['pvalue'] < 0.05)

                out[pred] = entry

        else:  # ols or robust-ols
            # use model.params, model.bse, model.tvalues, model.pvalues, model.conf_int()
            try:
                params = getattr(model, 'params', None)
                bse = getattr(model, 'bse', None)
                pvals = getattr(model, 'pvalues', None)
                tvals = getattr(model, 'tvalues', None)
                conf = None
                try:
                    conf = model.conf_int()
                except Exception:
                    conf = None
            except Exception:
                params = bse = pvals = tvals = conf = None

            for pred in predictors:
                entry = {}
                coef = None
                se = None
                pval = None
                stat = None
                ci_lo = None
                ci_hi = None

                try:
                    if params is not None:
                        coef = float(params[pred])
                except Exception:
                    coef = np.nan
                try:
                    if bse is not None:
                        se = float(bse[pred])
                except Exception:
                    se = np.nan
                try:
                    if pvals is not None:
                        pval = float(pvals[pred])
                except Exception:
                    pval = np.nan
                try:
                    if tvals is not None:
                        stat = float(tvals[pred])
                except Exception:
                    # compute from coef/se
                    try:
                        if (se is not None) and isfinite(se) and (se != 0):
                            stat = float(coef / se)
                    except Exception:
                        stat = None
                try:
                    if conf is not None:
                        # conf may be a DataFrame-like with rows indexed by param names
                        ci_lo = float(conf.loc[pred, 0]) if hasattr(conf, 'loc') else float(conf[pred][0])
                        ci_hi = float(conf.loc[pred, 1]) if hasattr(conf, 'loc') else float(conf[pred][1])
                    else:
                        if (se is not None) and isfinite(se) and isfinite(coef):
                            ci_lo = coef - 1.96 * se
                            ci_hi = coef + 1.96 * se
                except Exception:
                    ci_lo = ci_hi = np.nan

                entry['coef'] = None if coef is None or not isfinite(coef) else float(coef)
                entry['se'] = None if se is None or not isfinite(se) else float(se)
                entry['stat'] = None if stat is None or not isfinite(stat) else float(stat)
                entry['pvalue'] = None if pval is None or not isfinite(pval) else float(pval)
                entry['ci_lower'] = None if ci_lo is None or not isfinite(ci_lo) else float(ci_lo)
                entry['ci_upper'] = None if ci_hi is None or not isfinite(ci_hi) else float(ci_hi)
                try:
                    if entry['coef'] is not None:
                        entry['percent_change'] = float((np.exp(entry['coef']) - 1.0) * 100.0)
                    else:
                        entry['percent_change'] = None
                except Exception:
                    entry['percent_change'] = None

                entry['significant'] = (entry['pvalue'] is not None) and (entry['pvalue'] < 0.05)

                out[pred] = entry

        return out

    # Extract for mixedlm if present
    mixed_model = model_output.get('mixedlm', None)
    ols_model = model_output.get('ols_cluster', None)

    results['mixedlm'] = process_model(mixed_model, 'mixed')
    results['ols_cluster'] = process_model(ols_model, 'ols')

    # Build a concise conclusion for each predictor combining both models
    conclusion = {}
    for pred in predictors:
        concl = {}
        m = results['mixedlm'].get(pred, {})
        o = results['ols_cluster'].get(pred, {})
        # prefer mixed model for inference but report both
        concl['mixed_coef'] = m.get('coef')
        concl['mixed_pvalue'] = m.get('pvalue')
        concl['mixed_significant'] = m.get('significant')
        concl['ols_coef'] = o.get('coef')
        concl['ols_pvalue'] = o.get('pvalue')
        concl['ols_significant'] = o.get('significant')
        # simple textual summary
        summary_parts = []
        if m.get('coef') is not None:
            summary_parts.append(
                f"MixedLM: coef={m['coef']:.4f}, p={m['pvalue']:.3g}" if m.get('pvalue') is not None else
                f"MixedLM: coef={m['coef']:.4f}"
            )
        if o.get('coef') is not None:
            summary_parts.append(
                f"OLS(cluster): coef={o['coef']:.4f}, p={o['pvalue']:.3g}" if o.get('pvalue') is not None else
                f"OLS(cluster): coef={o['coef']:.4f}"
            )
        concl['summary'] = '; '.join(summary_parts) if summary_parts else "No estimate available"
        # overall significance if either model shows p<0.05
        concl['evidence_for_effect'] = bool((m.get('significant') is True) or (o.get('significant') is True))
        # direction based on mixed coef if available else ols
        direction_coef = m.get('coef') if m.get('coef') is not None else o.get('coef')
        if direction_coef is None:
            concl['direction'] = None
        elif direction_coef > 0:
            concl['direction'] = 'positive'
        elif direction_coef < 0:
            concl['direction'] = 'negative'
        else:
            concl['direction'] = 'no effect'
        conclusion[pred] = concl

    final = {
        "object": {
            "predictors": predictors,
            "models": results,
            "conclusion": conclusion
        },
        "description": (
            "For each predictor (age, sex_m, help_yes) this returns: coefficient (on log-efficiency scale), "
            "standard error, test statistic (z or t), p-value, 95% CI (approx), "
            "percent_change = (exp(coef)-1)*100 which is the approximate percent change in nuts-opened-per-second "
            "per unit increase in the predictor, and a boolean 'significant' flag (p < 0.05). "
            "The 'conclusion' section summarizes and indicates whether there is evidence for an effect "
            "based on either the mixed-effects model or the OLS with cluster-robust SEs."
        )
    }

    return final