def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs for 'beauty_c' and 'beauty_sq'
    from the provided model_output dict (expected keys: 'ols', 'ols_cluster', 'mixedlm').

    Returns a dict:
      - "object": a structured dict with per-model statistics and marginal effects at
                  beauty_c = 0, +1, -1 (units = 1 SD of the original beauty measure if that was used).
      - "description": a short plain-language interpretation that answers whether
                       instructor beauty affects teaching evaluations, referring to the
                       preferred model (cluster-robust OLS if available, otherwise mixedlm,
                       otherwise plain OLS).
    """
    def safe_get_stats(res, var):
        """Return dict with coef, se, pvalue, ci_lower, ci_upper for variable var from result object res.
           If any item is not available, set to None."""
        out = {'coef': None, 'se': None, 'pvalue': None, 'ci_lower': None, 'ci_upper': None}
        try:
            # coefficients
            params = getattr(res, 'params', None)
            if params is not None and var in params.index:
                out['coef'] = float(params[var])
        except Exception:
            pass
        try:
            bse = getattr(res, 'bse', None)
            if bse is not None and var in bse.index:
                out['se'] = float(bse[var])
        except Exception:
            pass
        try:
            pvals = getattr(res, 'pvalues', None)
            if pvals is not None and var in pvals.index:
                out['pvalue'] = float(pvals[var])
        except Exception:
            pass
        try:
            # conf_int may be a method or attribute
            if hasattr(res, 'conf_int'):
                ci = res.conf_int()
                # conf_int() returns DataFrame/array indexed like params
                if hasattr(ci, 'loc') and var in ci.index:
                    out['ci_lower'] = float(ci.loc[var][0])
                    out['ci_upper'] = float(ci.loc[var][1])
                else:
                    # try positional access if index alignment is different
                    try:
                        idx = list(params.index).index(var)
                        out['ci_lower'] = float(ci[idx, 0])
                        out['ci_upper'] = float(ci[idx, 1])
                    except Exception:
                        pass
        except Exception:
            pass
        return out

    results_summary = {}
    # iterate models present in model_output
    for key, res in (model_output or {}).items():
        # skip non-result entries (errors stored as strings)
        if not hasattr(res, 'params'):
            results_summary[key] = {'error': str(res)}
            continue
        stats_beauty = safe_get_stats(res, 'beauty_c')
        stats_beauty_sq = safe_get_stats(res, 'beauty_sq')

        # marginal effects for a few representative values of beauty_c (mean-centered):
        # effect = d(eval)/d(beauty_c) = beta1 + 2*beta2 * beauty_c
        def marginal_effect_at(v):
            b1 = stats_beauty['coef']
            b2 = stats_beauty_sq['coef']
            if b1 is None:
                return None
            if b2 is None:
                return {'marginal_effect': float(b1), 'details': 'no quadratic term available'}
            try:
                me = float(b1 + 2.0 * b2 * v)
                return {'marginal_effect': me, 'at_beauty_c': v}
            except Exception:
                return None

        me_at_0 = marginal_effect_at(0)   # effect at mean beauty
        me_at_plus1 = marginal_effect_at(1)
        me_at_minus1 = marginal_effect_at(-1)

        # significance flags
        sig_beauty = None
        sig_beauty_sq = None
        try:
            if stats_beauty['pvalue'] is not None:
                sig_beauty = float(stats_beauty['pvalue']) < 0.05
        except Exception:
            pass
        try:
            if stats_beauty_sq['pvalue'] is not None:
                sig_beauty_sq = float(stats_beauty_sq['pvalue']) < 0.05
        except Exception:
            pass

        results_summary[key] = {
            'beauty_c': stats_beauty,
            'beauty_sq': stats_beauty_sq,
            'marginal_effects': {
                'at_0': me_at_0,
                'at_plus_1': me_at_plus1,
                'at_minus_1': me_at_minus1
            },
            'significant': {
                'beauty_c_p_lt_0.05': sig_beauty,
                'beauty_sq_p_lt_0.05': sig_beauty_sq
            }
        }

    # Choose primary model for final verdict: prefer clustered OLS, then mixedlm, then plain ols
    primary_model = None
    for pref in ['ols_cluster', 'mixedlm', 'ols']:
        if pref in results_summary and 'error' not in results_summary[pref]:
            primary_model = pref
            break

    if primary_model is None:
        description = ("No usable fitted model objects found in model_output to draw conclusions.")
        return {'object': results_summary, 'description': description}

    prim_stats = results_summary[primary_model]
    # Interpret primary model results
    b = prim_stats['beauty_c']['coef']
    p = prim_stats['beauty_c']['pvalue']
    b_sq = prim_stats['beauty_sq']['coef']
    p_sq = prim_stats['beauty_sq']['pvalue']

    # Build interpretation text
    lines = []
    lines.append(f"Primary model used for inference: '{primary_model}'.")
    if b is None:
        lines.append("The primary model does not provide an estimate for 'beauty_c'.")
    else:
        lines.append(f"Estimated linear effect (beauty_c): coef = {b:.4g}"
                     + (f", p = {p:.3g}" if p is not None else ", p-value unavailable")
                     + (", statistically significant (p<0.05)." if (p is not None and p < 0.05) else (", not statistically significant." if p is not None else ".")))
    if b_sq is not None:
        lines.append(f"Quadratic term (beauty_sq): coef = {b_sq:.4g}"
                     + (f", p = {p_sq:.3g}" if p_sq is not None else ", p-value unavailable")
                     + (", statistically significant (p<0.05)." if (p_sq is not None and p_sq < 0.05) else (", not statistically significant." if p_sq is not None else ".")))
        lines.append("Marginal effect of beauty on evaluation = beta1 + 2*beta2*beauty_c. "
                     "At mean beauty (beauty_c=0) the effect equals the linear coef above.")
    else:
        lines.append("No quadratic effect estimated (or available); interpret the linear coef as the average change in evaluation per one-unit increase in mean-centered beauty.")

    # Add numeric marginal effects if available
    me0 = prim_stats['marginal_effects'].get('at_0')
    if me0 and 'marginal_effect' in me0:
        lines.append(f"Marginal effect at beauty_c = 0: {me0['marginal_effect']:.4g} rating points per unit of centered beauty.")
    me1 = prim_stats['marginal_effects'].get('at_plus_1')
    if me1 and 'marginal_effect' in me1:
        lines.append(f"Marginal effect at beauty_c = +1: {me1['marginal_effect']:.4g}.")
    me_1 = prim_stats['marginal_effects'].get('at_minus_1')
    if me_1 and 'marginal_effect' in me_1:
        lines.append(f"Marginal effect at beauty_c = -1: {me_1['marginal_effect']:.4g}.")

    # Final concise answer to the question
    if prim_stats['significant']['beauty_c_p_lt_0.05'] is True:
        lines.append("Conclusion (primary model): There is evidence that instructor physical attractiveness is associated with higher/lower student evaluation scores (the linear term is statistically significant).")
    elif prim_stats['significant']['beauty_c_p_lt_0.05'] is False:
        lines.append("Conclusion (primary model): No statistically significant association between instructor attractiveness and student evaluation scores was detected in the primary model.")
    else:
        lines.append("Conclusion (primary model): Cannot determine statistical significance for the attractiveness effect from the available primary-model outputs.")

    description = " ".join(lines)

    return {'object': results_summary, 'description': description}