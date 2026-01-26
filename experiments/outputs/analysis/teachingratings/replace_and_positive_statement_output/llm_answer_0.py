def extract_final_answer(model_output):
    """
    Extract coefficient, standard error, p-value, and 95% CI for 'beauty_z'
    from the provided model_output dict that should contain keys:
      - 'ols_model'      : statsmodels RegressionResultsWrapper
      - 'ols_clustered'  : statsmodels results object from get_robustcov_results(...)
      - 'mixedlm'        : statsmodels MixedLMResultsWrapper OR dict with 'error' if failed

    Returns:
      {
        "object": {
           "ols": { "coef": ..., "se": ..., "pvalue": ..., "ci95": [low, high] or None },
           "ols_clustered": { ... },
           "mixedlm": { ... },
           "final_conclusion": "Yes/No/Unclear ...",
           "summary_decision": { "n_models": 3, "n_significant_positive": X, "n_significant_negative": Y }
        },
        "description": "Brief interpretation of the coefficient(s) and final answer."
      }
    """
    import numpy as np

    def _extract_from_result(res, param_name='beauty_z'):
        """
        Try several common attributes to extract coef, se, pvalue, conf_int.
        Returns dict or None if extraction fails.
        """
        out = {"coef": None, "se": None, "pvalue": None, "ci95": None, "note": None}
        if res is None:
            out["note"] = "model result is None"
            return out

        # If res is an error dict (for mixedlm fallback), propagate
        if isinstance(res, dict) and 'error' in res:
            out["note"] = f"model error: {res.get('error')}"
            return out

        # Helper to try indexing by name or position
        def try_get_attr(container, attr):
            try:
                return getattr(container, attr)
            except Exception:
                return None

        # Try several places for params
        params = None
        try:
            # Many results have .params (Series) or .params (ndarray)
            params = try_get_attr(res, 'params') or try_get_attr(res, 'fe_params')
        except Exception:
            params = None

        # If params is a pandas Series or dict-like, try to index by name
        try:
            if params is not None:
                # If it's a pandas Series, this will work; if ndarray, fall through
                coef = params[param_name]
                out['coef'] = float(coef)
        except Exception:
            # If params is an ndarray and we can't find index, try to find by position using param names
            try:
                names = None
                # statsmodels sometimes stores param names in .model.exog_names or .param_names
                names = try_get_attr(res, 'model') and getattr(res.model, 'exog_names', None)
                if not names:
                    names = try_get_attr(res, 'param_names') or try_get_attr(res, 'bse') and getattr(res, 'bse').index
                if names:
                    if isinstance(names, (list, tuple)) and param_name in names:
                        pos = list(names).index(param_name)
                        out['coef'] = float(params[pos])
            except Exception:
                pass

        # Standard error
        try:
            # For many result objects .bse or .bse_fe exist
            bse = try_get_attr(res, 'bse') or try_get_attr(res, 'bse_fe')
            if bse is not None:
                try:
                    out['se'] = float(bse[param_name])
                except Exception:
                    # bse could be ndarray; try matching name positions as above
                    try:
                        names = try_get_attr(res, 'model') and getattr(res.model, 'exog_names', None)
                        if names and param_name in names:
                            pos = list(names).index(param_name)
                            out['se'] = float(bse[pos])
                    except Exception:
                        pass
        except Exception:
            pass

        # p-value
        try:
            pvalues = try_get_attr(res, 'pvalues') or try_get_attr(res, 'pvalues_fe')
            if pvalues is not None:
                try:
                    out['pvalue'] = float(pvalues[param_name])
                except Exception:
                    try:
                        names = try_get_attr(res, 'model') and getattr(res.model, 'exog_names', None)
                        if names and param_name in names:
                            pos = list(names).index(param_name)
                            out['pvalue'] = float(pvalues[pos])
                    except Exception:
                        pass
        except Exception:
            pass

        # 95% confidence interval
        try:
            ci = None
            # Many results have .conf_int() returning DataFrame/ndarray
            conf = try_get_attr(res, 'conf_int')
            if callable(conf):
                conf = conf()
            if conf is not None:
                try:
                    # conf could be DataFrame with index of param names
                    if hasattr(conf, 'loc') and param_name in conf.index:
                        row = conf.loc[param_name]
                        out['ci95'] = [float(row.iloc[0]), float(row.iloc[1])]
                    else:
                        # conf might be ndarray with ordering matching params
                        names = try_get_attr(res, 'model') and getattr(res.model, 'exog_names', None)
                        if names and param_name in names:
                            pos = list(names).index(param_name)
                            out['ci95'] = [float(conf[pos, 0]), float(conf[pos, 1])]
                except Exception:
                    pass
        except Exception:
            pass

        # If any of coef/se/pvalue are still None, try alternative attributes for MixedLM
        if out['coef'] is None:
            try:
                fe = try_get_attr(res, 'fe_params')
                if fe is not None and param_name in fe.index:
                    out['coef'] = float(fe[param_name])
            except Exception:
                pass

        if out['se'] is None:
            try:
                bse_fe = try_get_attr(res, 'bse_fe')
                if bse_fe is not None and param_name in bse_fe.index:
                    out['se'] = float(bse_fe[param_name])
            except Exception:
                pass

        if out['pvalue'] is None:
            # Some MixedLMResults lack pvalues; we can compute approximate pvalue using z = coef/se and normal dist
            try:
                if out['coef'] is not None and out['se'] is not None and out['se'] != 0:
                    z = out['coef'] / out['se']
                    # two-sided p-value from normal approximation
                    from math import erf, sqrt
                    # Use scipy not available; use normal cdf approx via erf
                    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
                    out['pvalue'] = float(p)
                    out['note'] = (out['note'] + '; p-value computed from z-stat') if out.get('note') else 'p-value computed from z-stat'
            except Exception:
                pass

        # Rounding numeric outputs for readability
        for k in ['coef', 'se', 'pvalue']:
            if out[k] is not None:
                try:
                    out[k] = float(np.round(out[k], 4))
                except Exception:
                    pass
        if out['ci95'] is not None:
            try:
                out['ci95'] = [float(np.round(out['ci95'][0], 4)), float(np.round(out['ci95'][1], 4))]
            except Exception:
                pass

        return out

    results = {}
    keys = ['ols_model', 'ols_clustered', 'mixedlm']
    readable_keys = {'ols_model': 'ols', 'ols_clustered': 'ols_clustered', 'mixedlm': 'mixedlm'}
    for k in keys:
        res = model_output.get(k)
        results[readable_keys[k]] = _extract_from_result(res, param_name='beauty_z')

    # Build a simple decision about significance and direction
    n_models = 0
    n_sig_pos = 0
    n_sig_neg = 0
    sig_alpha = 0.05
    for k, v in results.items():
        # If there's a numeric pvalue and coef, consider model counted
        if v.get('pvalue') is not None and v.get('coef') is not None:
            n_models += 1
            if v['pvalue'] < sig_alpha:
                if v['coef'] > 0:
                    n_sig_pos += 1
                elif v['coef'] < 0:
                    n_sig_neg += 1

    # Final conclusion: majority of models showing significant positive effect => "Yes"
    if n_sig_pos > n_sig_neg and n_sig_pos >= 1:
        final = (
            f"Yes — higher instructor physical attractiveness is associated with higher student evaluations. "
            f"{n_sig_pos} model(s) show a statistically significant positive effect (p < {sig_alpha})."
        )
    elif n_sig_neg > n_sig_pos and n_sig_neg >= 1:
        final = (
            f"Yes (negative) — higher beauty associated with lower evaluations in {n_sig_neg} model(s) (p < {sig_alpha})."
        )
    elif n_sig_pos == 0 and n_sig_neg == 0 and n_models > 0:
        final = (
            "No clear evidence — none of the models produced a statistically significant effect of beauty "
            f"at alpha={sig_alpha}."
        )
    else:
        final = "Unclear — mixed evidence across models."

    # Add interpretative note about units: beauty_z is standardized, so coef = change in eval per SD increase
    description = (
        "Extracted coefficient, standard error, p-value, and 95% CI for the predictor 'beauty_z' "
        "(instructor physical attractiveness standardized). The coefficient represents the expected change "
        "in the student course evaluation (eval) for a one standard-deviation increase in beauty. "
        "Confidence intervals give the plausible range for the effect. The 'final_conclusion' summarizes "
        "whether the effect is statistically significant (two-sided) in the fitted models."
    )

    output = {
        "object": {
            "models": results,
            "final_conclusion": final,
            "summary_decision": {
                "n_models_with_coef_and_pvalue": n_models,
                "n_significant_positive": n_sig_pos,
                "n_significant_negative": n_sig_neg,
                "alpha": sig_alpha
            }
        },
        "description": description
    }

    return output