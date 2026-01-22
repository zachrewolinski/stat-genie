def extract_final_answer(model_output):
    """
    Extract key statistics for the predictors of interest (age_c, sex_m, help_y)
    from the provided model_output dict (expected to contain 'mixedlm' and/or 'ols'
    fitted statsmodels result objects and optionally 'group_stats').

    Returns:
      {
        "object": {
            "mixedlm": { predictor: {coef, se, pvalue, ci_lower, ci_upper} or None, ... } or error message,
            "ols":    { predictor: {coef, se, pvalue, ci_lower, ci_upper} or None, ... } or error message,
            "group_stats": dict (if present)
        },
        "description": brief human-readable interpretation (which model used for primary inference
                        and summary lines for each predictor)
      }
    """
    preds = ['age_c', 'sex_m', 'help_y']
    out = {}
    # Helper to safely extract CI rows/entries
    def _get_ci(ci_obj, name, params_index):
        if ci_obj is None:
            return (None, None)
        try:
            # If CI supports .loc (DataFrame)
            lower = float(ci_obj.loc[name, 0])
            upper = float(ci_obj.loc[name, 1])
            return (lower, upper)
        except Exception:
            try:
                # If CI is array-like and in the same order as params_index
                idx = list(params_index).index(name)
                lower = float(ci_obj[idx, 0])
                upper = float(ci_obj[idx, 1])
                return (lower, upper)
            except Exception:
                return (None, None)

    # Extract from mixed model if present
    if 'mixedlm' in model_output:
        mm = model_output['mixedlm']
        try:
            # statsmodels MixedLMResults: fixed effects often in .fe_params (or .params)
            params = getattr(mm, 'fe_params', None) or getattr(mm, 'params', None)
            # standard errors for fixed effects
            bse = getattr(mm, 'bse_fe', None) or getattr(mm, 'bse', None)
            # p-values (may or may not be present)
            pvals = getattr(mm, 'pvalues', None)
            # confidence intervals
            try:
                ci = mm.conf_int()
            except Exception:
                ci = None

            mixed_info = {}
            for p in preds:
                if params is None or p not in params.index:
                    mixed_info[p] = None
                    continue
                coef = float(params[p])
                se = float(bse[p]) if (bse is not None and p in getattr(bse, 'index', list(params.index))) else None
                pval = float(pvals[p]) if (pvals is not None and p in getattr(pvals, 'index', list(params.index))) else None
                lower, upper = _get_ci(ci, p, params.index) if ci is not None else (None, None)
                mixed_info[p] = {'coef': coef, 'se': se, 'pvalue': pval, 'ci_lower': lower, 'ci_upper': upper}
            out['mixedlm'] = mixed_info
        except Exception as e:
            out['mixedlm_error'] = str(e)

    # Extract from OLS if present
    if 'ols' in model_output:
        ols = model_output['ols']
        try:
            params = ols.params
            bse = ols.bse
            pvals = ols.pvalues
            ci = ols.conf_int()
            ols_info = {}
            for p in preds:
                if p not in params.index:
                    ols_info[p] = None
                    continue
                coef = float(params[p])
                se = float(bse[p]) if p in bse.index else None
                pval = float(pvals[p]) if p in pvals.index else None
                lower = float(ci.loc[p, 0]) if (p in ci.index) else None
                upper = float(ci.loc[p, 1]) if (p in ci.index) else None
                ols_info[p] = {'coef': coef, 'se': se, 'pvalue': pval, 'ci_lower': lower, 'ci_upper': upper}
            out['ols'] = ols_info
        except Exception as e:
            out['ols_error'] = str(e)

    # Include group summaries if provided
    if 'group_stats' in model_output:
        try:
            gs = model_output['group_stats']
            # convert DataFrame to serializable dict of lists if necessary
            if hasattr(gs, 'to_dict'):
                out['group_stats'] = gs.to_dict(orient='list')
            else:
                out['group_stats'] = gs
        except Exception:
            out['group_stats'] = str(model_output['group_stats'])

    # Build a short interpretation string using mixed model as primary if available
    primary = None
    if 'mixedlm' in out and not out.get('mixedlm_error'):
        primary = 'mixedlm'
    elif 'ols' in out and not out.get('ols_error'):
        primary = 'ols'

    desc_lines = []
    if primary is None:
        desc_lines.append("No usable model results found to form an inference.")
    else:
        desc_lines.append(f"Primary inference based on the '{primary}' model results.")
        for p in preds:
            info = out.get(primary, {}).get(p) if isinstance(out.get(primary), dict) else None
            if info is None:
                desc_lines.append(f"- {p}: not estimated or not available in {primary}.")
            else:
                coef = info['coef']
                pval = info['pvalue']
                lower = info['ci_lower']
                upper = info['ci_upper']
                direction = "positive" if coef > 0 else ("negative" if coef < 0 else "no effect")
                if pval is None:
                    signif = "p-value not available"
                else:
                    signif = "statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)"
                desc_lines.append(
                    f"- {p}: coef = {coef:.4f}, {signif}, direction = {direction}, 95% CI = [{None if lower is None else round(lower,4)}, {None if upper is None else round(upper,4)}]."
                )

    description = " ".join(desc_lines)
    return {"object": out, "description": description}