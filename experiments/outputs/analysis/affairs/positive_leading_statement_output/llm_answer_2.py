def extract_final_answer(model_output):
    """
    Extracts statistics for the 'has_children' coefficient from the provided
    model_output (a dict of fitted model result objects and descriptive stats).

    Returns a dict with:
      - "object": dict of extracted numeric results per model plus group stats and t-test
      - "description": short interpretation summarizing direction, significance, and overall conclusion
    """
    import math
    from collections import Counter

    def _get_series_value(container, key_candidates):
        """
        Helper to fetch a value from a pandas Series-like container by trying
        multiple candidate key names and falling back to substring matches.
        Returns (value, matched_key) or (None, None).
        """
        try:
            # direct indexing if possible
            for k in key_candidates:
                try:
                    val = container[k]
                    return val, k
                except Exception:
                    continue
            # try substring match (avoid inflation params)
            try:
                keys = list(container.index)
                for k in keys:
                    kl = k.lower()
                    if 'infl' in kl or 'inflate' in kl:
                        continue
                    for cand in key_candidates:
                        if cand in k:
                            return container[k], k
            except Exception:
                pass
        except Exception:
            pass
        return None, None

    def extract_from_result(res, param_name='has_children'):
        """
        Extract coef, se, stat, pvalue, conf_int for param_name from a statsmodels result object.
        Also compute IRR = exp(coef) and IRR CI (useful for count models).
        Returns a dict with these fields (values or None).
        """
        out = {
            'coef': None,
            'se': None,
            'stat': None,
            'pvalue': None,
            'ci_lower': None,
            'ci_upper': None,
            'irr': None,
            'irr_ci_lower': None,
            'irr_ci_upper': None,
            'note': None
        }
        if res is None:
            out['note'] = 'result object is None'
            return out

        # 1) attempt to extract coefficient
        try:
            params = getattr(res, 'params', None)
            if params is not None:
                val, matched = _get_series_value(params, [param_name])
                if val is not None:
                    out['coef'] = float(val)
                else:
                    # fallback: try to find by model exog_names if available
                    try:
                        names = getattr(getattr(res, 'model', None), 'exog_names', None)
                        if names and param_name in names:
                            idx = names.index(param_name)
                            out['coef'] = float(params[idx])
                    except Exception:
                        pass
        except Exception as e:
            out['note'] = f'error extracting coef: {e}'

        # 2) standard error
        try:
            bse = getattr(res, 'bse', None)
            if bse is not None:
                val, matched = _get_series_value(bse, [param_name])
                if val is not None:
                    out['se'] = float(val)
                else:
                    try:
                        names = getattr(getattr(res, 'model', None), 'exog_names', None)
                        if names and param_name in names:
                            idx = names.index(param_name)
                            out['se'] = float(bse[idx])
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) stat and pvalue (stat may be t or z)
        try:
            pvals = getattr(res, 'pvalues', None)
            if pvals is not None:
                val, matched = _get_series_value(pvals, [param_name])
                if val is not None:
                    out['pvalue'] = float(val)
                else:
                    try:
                        names = getattr(getattr(res, 'model', None), 'exog_names', None)
                        if names and param_name in names:
                            idx = names.index(param_name)
                            out['pvalue'] = float(pvals[idx])
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            # try tvalues then zvalues
            stat_series = None
            if hasattr(res, 'tvalues'):
                stat_series = getattr(res, 'tvalues')
            elif hasattr(res, 'zvalues'):
                stat_series = getattr(res, 'zvalues')
            if stat_series is not None:
                val, matched = _get_series_value(stat_series, [param_name])
                if val is not None:
                    out['stat'] = float(val)
                else:
                    try:
                        names = getattr(getattr(res, 'model', None), 'exog_names', None)
                        if names and param_name in names:
                            idx = names.index(param_name)
                            out['stat'] = float(stat_series[idx])
                    except Exception:
                        pass
        except Exception:
            pass

        # 4) confidence interval
        try:
            ci = res.conf_int()
            # ci could be DataFrame with index names or ndarray in order of params
            if hasattr(ci, 'loc'):
                # DataFrame-like
                if param_name in ci.index:
                    lo, hi = ci.loc[param_name].tolist()
                    out['ci_lower'], out['ci_upper'] = float(lo), float(hi)
                else:
                    # attempt substring match in index (avoid inflation params)
                    matched_row = None
                    for idx in ci.index:
                        if param_name in str(idx) and 'infl' not in str(idx).lower():
                            matched_row = idx
                            break
                    if matched_row is not None:
                        lo, hi = ci.loc[matched_row].tolist()
                        out['ci_lower'], out['ci_upper'] = float(lo), float(hi)
                    else:
                        # fallback: try to align by ordering using model.exog_names
                        try:
                            names = getattr(getattr(res, 'model', None), 'exog_names', None)
                            if names and param_name in names:
                                i = names.index(param_name)
                                lo, hi = ci[i]
                                out['ci_lower'], out['ci_upper'] = float(lo), float(hi)
                        except Exception:
                            pass
            else:
                # ndarray-like: try to use exog_names for index mapping
                try:
                    names = getattr(getattr(res, 'model', None), 'exog_names', None)
                    if names and param_name in names:
                        i = names.index(param_name)
                        lo, hi = ci[i]
                        out['ci_lower'], out['ci_upper'] = float(lo), float(hi)
                except Exception:
                    pass
        except Exception:
            pass

        # 5) IRR and IRR CI (for count models)
        try:
            if out['coef'] is not None:
                out['irr'] = float(math.exp(out['coef']))
                if out['ci_lower'] is not None and out['ci_upper'] is not None:
                    out['irr_ci_lower'] = float(math.exp(out['ci_lower']))
                    out['irr_ci_upper'] = float(math.exp(out['ci_upper']))
        except Exception:
            pass

        return out

    final = {}
    # Extract from each model if present
    for model_name in ['negative_binomial', 'zero_inflated_negative_binomial', 'ols_robust']:
        if model_name in model_output and model_output[model_name] is not None:
            try:
                final[model_name] = extract_from_result(model_output[model_name], 'has_children')
            except Exception as e:
                final[model_name] = {'error': f'exception extracting stats: {e}'}
        else:
            final[model_name] = {'error': 'model not present in model_output'}

    # Attach descriptive group stats and Welch t-test if present
    final['group_stats'] = model_output.get('group_stats', None)
    final['welch_ttest'] = model_output.get('welch_ttest', None)

    # Build a concise interpretation across models
    interpretations = []
    sig_dirs = []
    for model_name in ['negative_binomial', 'zero_inflated_negative_binomial', 'ols_robust']:
        info = final.get(model_name, {})
        if info is None or 'error' in info:
            interpretations.append(f"{model_name}: no usable estimate.")
            continue
        coef = info.get('coef')
        pval = info.get('pvalue')
        irr = info.get('irr')
        if coef is None:
            interpretations.append(f"{model_name}: 'has_children' coefficient not found.")
            continue
        direction = 'decrease' if coef < 0 else 'increase' if coef > 0 else 'no change'
        sig = False
        if pval is not None:
            sig = (pval < 0.05)
        if model_name.startswith('negative') or model_name.startswith('zero'):
            if irr is not None:
                interpretations.append(
                    f"{model_name}: coef={coef:+.3f}, IRR={irr:.3f}, p={pval if pval is not None else 'NA'} -> "
                    f"{'significant' if sig else 'not significant'}; interpreted as a {direction} of {((irr-1)*100):+.1f}% in the rate of reported affairs."
                )
            else:
                interpretations.append(
                    f"{model_name}: coef={coef:+.3f}, p={pval if pval is not None else 'NA'} -> {'significant' if sig else 'not significant'}; direction: {direction}."
                )
        else:
            interpretations.append(
                f"{model_name}: coef={coef:+.3f} (change in mean affairs), p={pval if pval is not None else 'NA'} -> {'significant' if sig else 'not significant'}; direction: {direction}."
            )
        if sig:
            sig_dirs.append('decrease' if coef < 0 else 'increase' if coef > 0 else 'no change')

    # Use group-level Welch t-test as additional descriptive evidence
    wt = final.get('welch_ttest')
    gs = final.get('group_stats')
    if wt is not None:
        try:
            tstat = wt.get('statistic')
            pval = wt.get('pvalue')
            if gs:
                mean0 = gs.get('mean', {}).get(0)
                mean1 = gs.get('mean', {}).get(1)
                interpretations.append(f"Descriptive: parents mean affairs = {mean1:.3f}, non-parents mean = {mean0:.3f}; Welch t-test t={tstat:.3f}, p={pval:.3g}.")
                if pval is not None and pval < 0.05:
                    # direction by means
                    if mean1 is not None and mean0 is not None:
                        dir_text = 'increase' if mean1 > mean0 else 'decrease'
                        sig_dirs.append(dir_text)
            else:
                interpretations.append(f"Welch t-test: t={tstat:.3f}, p={pval:.3g}.")
        except Exception:
            pass

    # Aggregate final conclusion
    conclusion = "No decisive evidence either way."
    if len(sig_dirs) == 0:
        conclusion = "No consistent statistically significant evidence that having children decreases engagement in extramarital affairs."
    else:
        cnt = Counter(sig_dirs)
        most_common_dir, count = cnt.most_common(1)[0]
        if most_common_dir == 'decrease':
            conclusion = "Overall, the models with statistically significant results suggest having children is associated with a decrease in reported extramarital affairs."
        elif most_common_dir == 'increase':
            conclusion = "Overall, the models with statistically significant results suggest having children is associated with an increase in reported extramarital affairs."
        else:
            conclusion = "Statistically significant results are mixed in direction."

    description = "Extracted estimates for the 'has_children' coefficient from available models. " \
                  + "Model-by-model interpretations: " + " ".join(interpretations) + " Final conclusion: " + conclusion

    return {"object": final, "description": description}