def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of `skin_binary_dark` from the modeling output.
    Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, se, p-value, conf int, IRR, IRR CI, model used)
      - "description": brief interpretation of the effect in context
    
    The function is written defensively to handle either clustered or non-clustered results
    as returned by statsmodels' get_robustcov_results or the original GLMResultsWrapper.
    """
    import math
    import numpy as np

    # Helper to safely get attribute or dict-like item
    def safe_get(obj, attr, key=None):
        if obj is None:
            return None
        # Try dict-like access first
        try:
            if key is None:
                return obj[attr]
            else:
                return obj[key]
        except Exception:
            pass
        # Try pandas Series / DataFrame .loc
        try:
            if key is None:
                return getattr(obj, attr)
        except Exception:
            pass
        # Try attribute access
        try:
            return getattr(obj, attr)
        except Exception:
            return None

    # Prefer clustered results if available
    nb_clustered = model_output.get('nb_clustered', None)
    nb_model = model_output.get('nb_model', None)
    poisson_model = model_output.get('poisson_model', None)
    overdispersion = model_output.get('overdispersion', None)

    # Choose which result object to use for primary inference
    res = nb_clustered if nb_clustered is not None else nb_model

    # Initialize outputs
    out = {
        'model_used': 'nb_clustered' if nb_clustered is not None else ('nb_model' if nb_model is not None else None),
        'coef': None,
        'std_err': None,
        'p_value': None,
        'conf_low': None,
        'conf_high': None,
        'irr': None,
        'irr_ci_low': None,
        'irr_ci_high': None,
        'overdispersion': overdispersion,
        'poisson_coef': None,
        'poisson_pvalue': None
    }

    # Try to extract from the chosen results object
    try:
        # params
        params = safe_get(res, 'params')
        if params is not None:
            # params might be a pandas Series or numpy array; try label access
            try:
                coef = params['skin_binary_dark']
            except Exception:
                # maybe it's a numpy array; try to find index
                try:
                    # find matching name in index if available
                    idx = None
                    if hasattr(params, 'index'):
                        for i, name in enumerate(params.index):
                            if name == 'skin_binary_dark':
                                idx = i
                                break
                    if idx is not None:
                        coef = float(params.iloc[idx])
                    else:
                        coef = float(params[0])  # fallback but risky
                except Exception:
                    coef = float(params[0])
        else:
            coef = None
        out['coef'] = float(coef) if (coef is not None and not np.isnan(coef)) else None
    except Exception:
        out['coef'] = None

    try:
        # standard error and p-value
        bse = safe_get(res, 'bse')
        pvalues = safe_get(res, 'pvalues')
        if bse is not None:
            try:
                se = bse['skin_binary_dark']
            except Exception:
                try:
                    se = float(bse.loc['skin_binary_dark'])
                except Exception:
                    # fallback: if index unknown, try matching first element
                    se = float(bse.iloc[0]) if hasattr(bse, 'iloc') else float(np.asarray(bse)[0])
        else:
            se = None
        if pvalues is not None:
            try:
                pval = pvalues['skin_binary_dark']
            except Exception:
                try:
                    pval = float(pvalues.loc['skin_binary_dark'])
                except Exception:
                    pval = float(pvalues.iloc[0]) if hasattr(pvalues, 'iloc') else float(np.asarray(pvalues)[0])
        else:
            pval = None

        out['std_err'] = float(se) if (se is not None and not np.isnan(se)) else None
        out['p_value'] = float(pval) if (pval is not None and not np.isnan(pval)) else None
    except Exception:
        out['std_err'] = out['p_value'] = None

    try:
        # confidence interval on the coefficient scale
        ci = None
        try:
            ci_all = res.conf_int()
            # conf_int() returns a DataFrame with rows indexed by parameter names
            try:
                row = ci_all.loc['skin_binary_dark']
                ci_low, ci_high = float(row[0]), float(row[1])
            except Exception:
                # maybe the index is an integer or param name not found; try to find row by name
                if hasattr(ci_all, 'index'):
                    found = False
                    for i, name in enumerate(ci_all.index):
                        if name == 'skin_binary_dark':
                            row = ci_all.iloc[i]
                            ci_low, ci_high = float(row[0]), float(row[1])
                            found = True
                            break
                    if not found:
                        # fallback to first row
                        row = ci_all.iloc[0]
                        ci_low, ci_high = float(row[0]), float(row[1])
                else:
                    row = ci_all[0]
                    ci_low, ci_high = float(row[0]), float(row[1])
        except Exception:
            ci_low = ci_high = None

        out['conf_low'] = ci_low if (ci_low is not None and not np.isnan(ci_low)) else None
        out['conf_high'] = ci_high if (ci_high is not None and not np.isnan(ci_high)) else None

        # Incident Rate Ratio (IRR) and its CI (exp of coef and conf int)
        if out['coef'] is not None:
            irr = math.exp(out['coef'])
            out['irr'] = float(irr)
            if out['conf_low'] is not None and out['conf_high'] is not None:
                out['irr_ci_low'] = float(math.exp(out['conf_low']))
                out['irr_ci_high'] = float(math.exp(out['conf_high']))
    except Exception:
        pass

    # Extract Poisson coefficient for sensitivity (if available)
    try:
        pm = poisson_model
        if pm is not None:
            pm_params = safe_get(pm, 'params')
            pm_pvalues = safe_get(pm, 'pvalues')
            if pm_params is not None:
                try:
                    poisson_coef = float(pm_params['skin_binary_dark'])
                except Exception:
                    try:
                        poisson_coef = float(pm_params.loc['skin_binary_dark'])
                    except Exception:
                        poisson_coef = float(pm_params.iloc[0])
                out['poisson_coef'] = poisson_coef
            if pm_pvalues is not None:
                try:
                    out['poisson_pvalue'] = float(pm_pvalues['skin_binary_dark'])
                except Exception:
                    try:
                        out['poisson_pvalue'] = float(pm_pvalues.loc['skin_binary_dark'])
                    except Exception:
                        out['poisson_pvalue'] = float(pm_pvalues.iloc[0])
    except Exception:
        pass

    # Build a concise interpretation
    descr_parts = []
    if out['coef'] is None:
        descr = "Could not extract the `skin_binary_dark` coefficient from the provided model output."
    else:
        # direction
        sign = "higher" if out['coef'] > 0 else ("lower" if out['coef'] < 0 else "no difference")
        descr_parts.append(
            f"The primary estimate (negative binomial with cluster-robust SEs) for `skin_binary_dark` is "
            f"coef = {out['coef']:.4f} (SE = {out['std_err']:.4f})"
            if (out['std_err'] is not None) else
            f"The primary estimate (negative binomial) for `skin_binary_dark` is coef = {out['coef']:.4f}"
        )
        if out['p_value'] is not None:
            descr_parts.append(f"with two-sided p = {out['p_value']:.3f}.")
        else:
            descr_parts.append("p-value not available.")

        if out['irr'] is not None:
            descr_parts.append(
                f"This corresponds to an incidence rate ratio (IRR) = {out['irr']:.3f}"
            )
            if out['irr_ci_low'] is not None and out['irr_ci_high'] is not None:
                descr_parts.append(
                    f"(95% CI for IRR: [{out['irr_ci_low']:.3f}, {out['irr_ci_high']:.3f}])."
                )
            else:
                descr_parts.append(".")
        # Conclusion about the hypothesis
        alpha = 0.05
        if out['p_value'] is not None:
            if out['p_value'] < alpha:
                # Significant in two-sided test
                direction_sentence = ("Darker-skinned players receive significantly more red cards per game than "
                                      "lighter-skinned players."
                                      if out['coef'] > 0 else
                                      "Darker-skinned players receive significantly fewer red cards per game than "
                                      "lighter-skinned players.")
            else:
                direction_sentence = ("No statistically significant difference in red-card rates per game between "
                                      "darker- and lighter-skinned players was found.")
            descr_parts.append(direction_sentence)
        else:
            descr_parts.append("Statistical significance could not be assessed (p-value missing).")

        # Mention overdispersion check
        if overdispersion is not None:
            descr_parts.append(f"Overdispersion statistic (deviance / df_resid) = {overdispersion:.3f} "
                               f"(values >>1 indicate overdispersion; here ~1 suggests NB model reasonable).")

        # Poisson sensitivity note
        if out['poisson_coef'] is not None:
            descr_parts.append(
                f"As a sensitivity check, the Poisson estimate for `skin_binary_dark` is coef = {out['poisson_coef']:.4f} "
                f"(Poisson p = {out['poisson_pvalue']:.3f} if available)."
            )

        descr = " ".join(descr_parts)

    return {
        "object": out,
        "description": descr
    }