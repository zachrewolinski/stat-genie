def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for:
      - the main effect of reader_view (effect for dyslexia_bin = 0)
      - the interaction term reader_view:dyslexia_bin
      - the marginal effect of reader_view for dyslexia_bin = 1 (reader_view + interaction)
    Returns a dict with keys:
      - "object": dict of numeric results
      - "description": short interpretation about whether Reader View improves reading speed,
                       overall and specifically for readers with dyslexia (based on p < 0.05).
    The function is defensive to handle slight variations in parameter naming.
    """
    import numpy as np

    res = model_output  # alias

    params = res.params
    pvals = res.pvalues
    try:
        conf = res.conf_int()
    except Exception:
        # fall back to manual CI using params +/- t*se
        conf = None
    cov = res.cov_params()

    # Helper to find parameter name robustly
    def find_param(containing=None, exact=None, exclude_contains=None):
        names = list(params.index.astype(str))
        if exact is not None:
            if exact in names:
                return exact
            # sometimes exact may appear with whitespace/encoding differences; try match
            for n in names:
                if n.strip() == exact:
                    return n
            return None
        if containing is not None:
            candidates = [n for n in names if containing in n]
            if exclude_contains:
                candidates = [n for n in candidates if exclude_contains not in n]
            # prefer a candidate without ':' (i.e., main effect, not interaction)
            if len(candidates) > 1:
                for c in candidates:
                    if ':' not in c:
                        return c
            return candidates[0] if candidates else None
        return None

    # try common names
    reader_name = find_param(exact='reader_view') or find_param(containing='reader_view', exclude_contains=':')
    dys_name = find_param(exact='dyslexia_bin') or find_param(containing='dyslexia_bin', exclude_contains=':')
    # interaction: any param that contains both substrings
    inter_name = None
    names = list(params.index.astype(str))
    for n in names:
        if ('reader_view' in n) and ('dyslexia_bin' in n):
            inter_name = n
            break

    result_obj = {}
    notes = []

    # function to assemble metrics for a given param name (single coefficient)
    def metrics_for_param(name):
        if name is None or name not in params.index:
            return None
        coef = float(params[name])
        se = float(np.sqrt(cov.loc[name, name])) if name in cov.index else float(np.nan)
        # confidence interval
        if conf is not None and name in conf.index:
            ci_low, ci_high = float(conf.loc[name, 0]), float(conf.loc[name, 1])
        else:
            # approximate using t with df_resid if possible
            try:
                df = float(res.df_resid)
                # try scipy t
                from scipy import stats
                tcrit = stats.t.ppf(0.975, df)
                ci_low = coef - tcrit * se
                ci_high = coef + tcrit * se
            except Exception:
                # normal approx
                z = 1.96
                ci_low = coef - z * se
                ci_high = coef + z * se
        pval = float(pvals[name]) if name in pvals.index else float('nan')
        return {'coef': coef, 'se': se, 'p': pval, 'ci_lower': ci_low, 'ci_upper': ci_high, 'name': name}

    # get metrics for main reader_view and interaction
    reader_metrics = metrics_for_param(reader_name)
    inter_metrics = metrics_for_param(inter_name)

    result_obj['reader_view_main_effect'] = reader_metrics
    result_obj['interaction_effect'] = inter_metrics

    # compute marginal effect of reader_view when dyslexia_bin = 1:
    # effect = coef(reader_view) + coef(interaction)
    def linear_combination(coef_terms):
        # coef_terms: dict term_name -> multiplier (e.g., {'reader_view':1, 'reader_view:dyslexia_bin':1})
        available = [t for t in coef_terms.keys() if t in params.index]
        if not available:
            return None
        est = 0.0
        # variance = a' V a
        a = np.zeros(len(params))
        index_map = {name: i for i, name in enumerate(params.index.astype(str))}
        for t, mult in coef_terms.items():
            if t in index_map:
                est += mult * float(params[t])
                a[index_map[t]] = mult
        # covariance matrix as ndarray aligned with params index
        try:
            V = cov.reindex(index=params.index, columns=params.index).values
            var = float(a @ V @ a)
            se = float(np.sqrt(var))
        except Exception:
            se = float('nan')
        # compute p-value using t-distribution with df_resid if possible, else normal approx
        if se == 0 or np.isnan(se):
            tstat = float('nan')
            pval = float('nan')
        else:
            tstat = est / se
            try:
                from scipy import stats
                df = float(res.df_resid)
                pval = 2.0 * (1.0 - stats.t.cdf(abs(tstat), df))
            except Exception:
                # normal approx
                from math import erf, sqrt
                import math
                # two-sided using normal
                pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / sqrt(2.0))))
        # confidence interval
        try:
            df = float(res.df_resid)
            from scipy import stats
            tcrit = stats.t.ppf(0.975, df)
            ci_low = est - tcrit * se
            ci_high = est + tcrit * se
        except Exception:
            z = 1.96
            ci_low = est - z * se
            ci_high = est + z * se
        return {'coef': float(est), 'se': float(se), 't': float(tstat), 'p': float(pval), 'ci_lower': float(ci_low), 'ci_upper': float(ci_high), 'terms': coef_terms}

    # marginal effect for dyslexia_bin = 0 is simply the main reader_view coefficient
    if reader_metrics is not None:
        marginal_dys0 = {'coef': reader_metrics['coef'], 'se': reader_metrics['se'], 'p': reader_metrics['p'],
                         'ci_lower': reader_metrics['ci_lower'], 'ci_upper': reader_metrics['ci_upper'], 'name': reader_metrics['name']}
    else:
        marginal_dys0 = None

    # marginal for dyslexia_bin = 1
    if reader_name is not None and inter_name is not None:
        marginal_dys1 = linear_combination({reader_name: 1.0, inter_name: 1.0})
    elif reader_name is not None and inter_name is None:
        # no interaction found; effect is same as main
        marginal_dys1 = marginal_dys0
    else:
        marginal_dys1 = None

    result_obj['marginal_effect_dyslexia_0'] = marginal_dys0
    result_obj['marginal_effect_dyslexia_1'] = marginal_dys1

    # Simple interpretation: Does Reader View improve reading speed?
    def interpret_effect(effect_dict, label):
        if effect_dict is None:
            return f"No estimate available for {label}."
        coef = effect_dict.get('coef', float('nan'))
        p = effect_dict.get('p', float('nan'))
        sig = (not np.isnan(p)) and (p < 0.05)
        direction = 'increase' if coef > 0 else ('decrease' if coef < 0 else 'no change')
        return f"For {label}: estimated effect = {coef:.3f} wpm, 95% CI [{effect_dict.get('ci_lower', float('nan')):.3f}, {effect_dict.get('ci_upper', float('nan')):.3f}], p = {p:.3g}. This indicates a {direction} in reading speed. {'Statistically significant (p<0.05).' if sig else 'Not statistically significant.'}"

    overall_statement = interpret_effect(marginal_dys0, "reader_view effect for non-dyslexic readers (dyslexia_bin=0)")
    dyslexic_statement = interpret_effect(marginal_dys1, "reader_view effect for dyslexic readers (dyslexia_bin=1)")
    interaction_statement = interpret_effect(inter_metrics, "interaction term (reader_view:dyslexia_bin)")

    # Final yes/no decision for question "Does Reader View improve reading speed for individuals with dyslexia?"
    decision = None
    if marginal_dys1 is None:
        decision = "Unable to determine (marginal effect for dyslexic readers not estimable from the model output)."
    else:
        coef = marginal_dys1.get('coef', 0.0)
        p = marginal_dys1.get('p', 1.0)
        if (not np.isnan(p)) and (p < 0.05) and (coef > 0):
            decision = "Yes — Reader View appears to improve reading speed for readers with dyslexia (statistically significant positive effect)."
        elif (not np.isnan(p)) and (p < 0.05) and (coef <= 0):
            decision = "No — Reader View appears to reduce (or not increase) reading speed for readers with dyslexia (statistically significant non-positive effect)."
        else:
            decision = "No (not statistically significant) — the model does not provide evidence that Reader View improves reading speed for readers with dyslexia."

    description = (
        "Extracted model estimates relevant to the question.\n"
        + overall_statement + "\n"
        + dyslexic_statement + "\n"
        + interaction_statement + "\n\n"
        + "Final conclusion regarding whether Reader View improves reading speed for individuals with dyslexia:\n"
        + decision
    )

    return {"object": result_obj, "description": description}