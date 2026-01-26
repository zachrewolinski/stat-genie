def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'Children' from OLS and ZINB model outputs.

    Returns a dictionary with:
      - "object": a dict containing numeric estimates, SEs, p-values and 95% CIs for the
                  'Children' coefficient from the OLS, and from both parts of the ZINB
                  (count model and inflation model).
      - "description": a short interpretation stating whether having children appears
                       to decrease engagement in extramarital affairs, based on the
                       direction and significance of the coefficients.

    The function is robust to common naming conventions used by statsmodels for zero-inflated
    parameter names (e.g., 'Children' and 'inflate_Children' or similar).
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'ols_results' and 'zinb_results'.")

    if 'ols_results' not in model_output or 'zinb_results' not in model_output:
        raise KeyError("model_output must contain keys 'ols_results' and 'zinb_results'.")

    ols = model_output['ols_results']
    zinb = model_output['zinb_results']

    result_obj = {}
    description_lines = []

    # Helper to safely extract parameter stats from a results object
    def _extract_from_result(res, param_name):
        # res: results wrapper (has .params, .bse, .pvalues, .conf_int())
        params = getattr(res, 'params', None)
        if params is None:
            return None
        if param_name not in params.index:
            return None
        coef = float(params[param_name])
        se = float(res.bse[param_name]) if hasattr(res, 'bse') and param_name in res.bse.index else None
        pval = float(res.pvalues[param_name]) if hasattr(res, 'pvalues') and param_name in res.pvalues.index else None
        ci_df = res.conf_int() if hasattr(res, 'conf_int') else None
        ci = list(map(float, ci_df.loc[param_name].values)) if (ci_df is not None and param_name in ci_df.index) else None
        return {'coef': coef, 'se': se, 'pval': pval, '95%_CI': ci}

    # 1) OLS: extract 'Children'
    ols_stats = _extract_from_result(ols, 'Children')
    if ols_stats is None:
        # Try alternative capitalization / prefix
        for name in ols.params.index:
            if 'children' in name.lower():
                ols_stats = _extract_from_result(ols, name)
                break

    if ols_stats is None:
        raise KeyError("Could not find a 'Children' coefficient in the OLS results.")

    result_obj['ols_children'] = ols_stats

    # Interpret OLS briefly
    ols_dir = 'negative' if ols_stats['coef'] < 0 else 'positive' if ols_stats['coef'] > 0 else 'zero'
    ols_sig = (ols_stats['pval'] is not None) and (ols_stats['pval'] < 0.05)
    description_lines.append(
        f"OLS: Children coef = {ols_stats['coef']:.4g} (SE={ols_stats['se']:.4g}, p={ols_stats['pval']:.4g}). "
        f"Direction: {ols_dir}. {'Statistically significant (p<0.05).' if ols_sig else 'Not statistically significant.'}"
    )

    # 2) ZINB: need to find both count and inflation coefficients for Children.
    zinb_params = zinb.params
    zinb_pvalues = getattr(zinb, 'pvalues', None)
    zinb_bse = getattr(zinb, 'bse', None)
    zinb_ci = zinb.conf_int() if hasattr(zinb, 'conf_int') else None

    # Find parameter names that refer to Children
    children_param_names = [n for n in zinb_params.index if 'children' in n.lower()]
    # Typical names: 'Children' (count), 'inflate_Children' or 'Children_infl' etc.
    count_name = None
    infl_name = None
    for n in children_param_names:
        if 'inflate' in n.lower() or 'infl' in n.lower() or 'zero' in n.lower():
            infl_name = n
        else:
            # If there are two names and one has a prefix/suffix indicating inflation, prefer that as infl.
            # Otherwise assume first is count if we haven't assigned count_name yet.
            if count_name is None:
                count_name = n

    # If not found by the above heuristic, try splitting params index into two blocks using model metadata
    if count_name is None or infl_name is None:
        # Try to use model attribute lists if available
        model = getattr(zinb, 'model', None)
        if model is not None:
            exog_names = getattr(model, 'exog_names', None)
            exog_infl_names = getattr(model, 'exog_infl_names', None)
            if exog_names is not None:
                for n in exog_names:
                    if 'children' in n.lower():
                        count_name = n
            if exog_infl_names is not None:
                for n in exog_infl_names:
                    if 'children' in n.lower():
                        # statsmodels may prefix inflation names when stored in params; find matching param
                        # find a param in zinb_params.index that contains n (or its basename)
                        matches = [p for p in zinb_params.index if n.lower() in p.lower()]
                        if matches:
                            infl_name = matches[0]

    # If still missing, fall back to any param name that contains 'children' and assign heuristically:
    if count_name is None and len(children_param_names) > 0:
        count_name = children_param_names[0]
        if len(children_param_names) > 1 and infl_name is None:
            infl_name = children_param_names[1]

    # Extract stats if names found
    zinb_count_stats = None
    zinb_infl_stats = None
    if count_name is not None:
        zinb_count_stats = {
            'name': count_name,
            'coef': float(zinb_params[count_name]),
            'se': float(zinb_bse[count_name]) if (zinb_bse is not None and count_name in zinb_bse.index) else None,
            'pval': float(zinb_pvalues[count_name]) if (zinb_pvalues is not None and count_name in zinb_pvalues.index) else None,
            '95%_CI': list(map(float, zinb_ci.loc[count_name].values)) if (zinb_ci is not None and count_name in zinb_ci.index) else None,
            'interpretation': 'count (log link)'
        }
        result_obj['zinb_count_children'] = zinb_count_stats

    if infl_name is not None:
        zinb_infl_stats = {
            'name': infl_name,
            'coef': float(zinb_params[infl_name]),
            'se': float(zinb_bse[infl_name]) if (zinb_bse is not None and infl_name in zinb_bse.index) else None,
            'pval': float(zinb_pvalues[infl_name]) if (zinb_pvalues is not None and infl_name in zinb_pvalues.index) else None,
            '95%_CI': list(map(float, zinb_ci.loc[infl_name].values)) if (zinb_ci is not None and infl_name in zinb_ci.index) else None,
            'interpretation': 'inflation (logit of being an excess-zero)'
        }
        result_obj['zinb_inflation_children'] = zinb_infl_stats

    if zinb_count_stats is None and zinb_infl_stats is None:
        raise KeyError("Could not locate 'Children' parameters in the ZINB results.")

    # Interpret ZINB:
    # - For count coef: negative -> lower expected counts (fewer affairs) among those with children (for the non-structural-zero group)
    # - For inflation coef: positive -> higher probability of being a structural zero (i.e., certain no-affair), which also means children -> fewer affairs
    if zinb_count_stats is not None:
        dir_count = 'negative' if zinb_count_stats['coef'] < 0 else 'positive' if zinb_count_stats['coef'] > 0 else 'zero'
        sig_count = (zinb_count_stats['pval'] is not None) and (zinb_count_stats['pval'] < 0.05)
        description_lines.append(
            f"ZINB (count): {zinb_count_stats['name']} coef = {zinb_count_stats['coef']:.4g} "
            f"(SE={zinb_count_stats['se']:.4g}, p={zinb_count_stats['pval']:.4g}). "
            f"Direction: {dir_count}. {'Significant.' if sig_count else 'Not significant.'}"
        )
    if zinb_infl_stats is not None:
        dir_infl = 'positive' if zinb_infl_stats['coef'] > 0 else 'negative' if zinb_infl_stats['coef'] < 0 else 'zero'
        sig_infl = (zinb_infl_stats['pval'] is not None) and (zinb_infl_stats['pval'] < 0.05)
        description_lines.append(
            f"ZINB (inflation): {zinb_infl_stats['name']} coef = {zinb_infl_stats['coef']:.4g} "
            f"(SE={zinb_infl_stats['se']:.4g}, p={zinb_infl_stats['pval']:.4g}). "
            f"Direction: {dir_infl}. {'Significant.' if sig_infl else 'Not significant.'}"
        )

    # Synthesize a simple conclusion about whether children decrease engagement in affairs.
    # Use ZINB evidence primarily (more appropriate for count-with-zeros), supported by OLS.
    evidence_count = None
    evidence_infl = None
    if zinb_count_stats is not None:
        evidence_count = (zinb_count_stats['coef'] < 0) and ((zinb_count_stats['pval'] is not None) and zinb_count_stats['pval'] < 0.05)
    if zinb_infl_stats is not None:
        # positive inflation coef that is significant indicates higher structural-zero probability (fewer affairs)
        evidence_infl = (zinb_infl_stats['coef'] > 0) and ((zinb_infl_stats['pval'] is not None) and zinb_infl_stats['pval'] < 0.05)

    # Combine
    if evidence_count or evidence_infl:
        conclusion = "Evidence that having children is associated with decreased engagement in extramarital affairs."
        # note if both significant
        if evidence_count and evidence_infl:
            conclusion += " Both the count and inflation parts of the ZINB indicate fewer affairs among those with children."
        elif evidence_count:
            conclusion += " The count part (expected number of affairs) is significantly lower for those with children."
        else:
            conclusion += " The inflation part (probability of being a certain zero/no-affair) is significantly higher for those with children."
    else:
        # If neither is significant but directions point to decrease, say 'weak' or 'no strong evidence'
        # Check directions
        dir_count_flag = (zinb_count_stats is not None) and (zinb_count_stats['coef'] < 0)
        dir_infl_flag = (zinb_infl_stats is not None) and (zinb_infl_stats['coef'] > 0)
        if (dir_count_flag or dir_infl_flag) and not (evidence_count or evidence_infl):
            conclusion = ("Point estimates from ZINB suggest children may be associated with fewer affairs "
                          " (negative count coef and/or positive inflation coef), but these estimates are not statistically significant.")
        else:
            # Use OLS as tie-breaker: if OLS significant negative and ZINB not significant, mention mixed evidence.
            ols_negative_sig = (ols_stats['coef'] < 0) and (ols_stats['pval'] is not None) and (ols_stats['pval'] < 0.05)
            if ols_negative_sig:
                conclusion = ("Mixed evidence: OLS indicates a statistically significant negative association between children and affairs, "
                              "but ZINB (more appropriate for this outcome) does not show significant effects.")
            else:
                conclusion = ("No strong evidence that having children decreases engagement in extramarital affairs in these models. "
                              "Point estimates are not consistently significant.")

    description_lines.append("Conclusion: " + conclusion)

    return {
        "object": result_obj,
        "description": " ".join(description_lines)
    }