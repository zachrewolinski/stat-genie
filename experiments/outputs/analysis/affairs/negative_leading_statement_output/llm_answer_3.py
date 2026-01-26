def extract_final_answer(model_output):
    """
    Extracts the effect of 'children_yes' from the provided model_output dict.
    Returns a dictionary with:
      - "object": a dict with extracted numeric results (model used, count coef & p,
                  inflation coef & p when available, IRR for count coef, and means_by_children)
      - "description": a brief human-readable interpretation of what the numbers imply
    The function tries, in order:
      1) to use the preferred model named in model_output['summary']['preferred_model'],
      2) then to fall back to available models in the order ['zinb','zip','ols'].
    It prefers a count-equation coefficient (non-inflation) when available; if only an
    inflation-term for children exists, it returns that and explains its meaning.
    """
    import math
    import numpy as np

    # Helpers
    def is_finite(x):
        try:
            return (x is not None) and (not (isinstance(x, float) and math.isnan(x))) and np.isfinite(x)
        except Exception:
            return False

    def find_children_params(params_index):
        """
        Given an iterable/list of parameter names, return two lists:
         - count_names: names likely referring to the count equation (prefer those without 'infl'/'inflate')
         - infl_names: names likely referring to the inflation equation (contain 'infl'/'inflate' or start with 'inflate')
        """
        count_names = []
        infl_names = []
        for name in params_index:
            lname = str(name).lower()
            if 'children' not in lname:
                continue
            if 'infl' in lname or 'inflate' in lname or lname.startswith('inflate') or lname.startswith('inflate.'):
                infl_names.append(name)
            else:
                count_names.append(name)
        return count_names, infl_names

    summary = model_output.get('summary', {}) if isinstance(model_output, dict) else {}
    means_by_children = summary.get('means_by_children', None)

    # Determine candidate model order
    tried = []
    preferred = summary.get('preferred_model') if isinstance(summary, dict) else None
    candidates = []
    if preferred:
        candidates.append(preferred)
    for c in ['zinb', 'zip', 'ols']:
        if c not in candidates:
            candidates.append(c)

    # Search models
    result = {
        'model_used': None,
        'count_coef': None,
        'count_pvalue': None,
        'count_IRR': None,            # exp(coef)
        'infl_coef': None,
        'infl_pvalue': None,
        'means_by_children': means_by_children
    }

    for mname in candidates:
        fitted = model_output.get(mname)
        if fitted is None:
            continue
        # try to get params and pvalues
        params = getattr(fitted, 'params', None)
        pvalues = getattr(fitted, 'pvalues', None)
        # Some wrappers might store params as numpy array with 'param_names' attribute; handle common case
        if params is None:
            # try summary dict fallback
            break
        # params should be a pandas Series or similar with an index of names
        try:
            index = list(params.index)
        except Exception:
            # If params has no index, skip this model
            continue

        count_names, infl_names = find_children_params(index)

        chosen_count = None
        chosen_infl = None
        if count_names:
            chosen_count = count_names[0]
        if infl_names:
            chosen_infl = infl_names[0]

        # If no count name but an inflation name exists, we'll still record the inflation effect.
        if chosen_count is None and chosen_infl is None:
            # nothing found in this model
            continue

        # Extract numeric values if available
        def safe_get(series, key):
            try:
                val = series[key]
                # convert numpy types to python floats
                return float(val)
            except Exception:
                return None

        count_coef = safe_get(params, chosen_count) if chosen_count is not None else None
        infl_coef = safe_get(params, chosen_infl) if chosen_infl is not None else None

        count_p = safe_get(pvalues, chosen_count) if (chosen_count is not None and pvalues is not None) else None
        infl_p = safe_get(pvalues, chosen_infl) if (chosen_infl is not None and pvalues is not None) else None

        # Check finite
        if not is_finite(count_coef):
            count_coef = None
            count_p = None
        if not is_finite(infl_coef):
            infl_coef = None
            infl_p = None

        # If no finite values in this model, try next
        if count_coef is None and infl_coef is None:
            continue

        # Fill result and break
        result['model_used'] = mname
        result['count_coef'] = count_coef
        result['count_pvalue'] = count_p
        result['infl_coef'] = infl_coef
        result['infl_pvalue'] = infl_p
        if count_coef is not None:
            try:
                result['count_IRR'] = float(np.exp(count_coef))
            except Exception:
                result['count_IRR'] = None
        break

    # If still nothing found, attempt to use the pre-computed summary children values if present
    if result['model_used'] is None and summary:
        # summary might have children_coef/pvalue even if nan; check finiteness
        sc = summary.get('children_coef', None)
        sp = summary.get('children_pvalue', None)
        if is_finite(sc):
            result['model_used'] = summary.get('preferred_model', None) or 'summary'
            result['count_coef'] = float(sc)
            result['count_pvalue'] = float(sp) if is_finite(sp) else None
            try:
                result['count_IRR'] = float(np.exp(sc))
            except Exception:
                result['count_IRR'] = None

    # Build description
    desc_parts = []
    if result['model_used'] is None:
        # No model-level coefficient available; fall back to reporting raw means if available
        if means_by_children:
            mean0 = means_by_children.get('mean', {}).get(0)
            mean1 = means_by_children.get('mean', {}).get(1)
            desc_parts.append("No usable coefficient for 'children_yes' could be extracted from the fitted models.")
            if (mean0 is not None) and (mean1 is not None):
                desc_parts.append(
                    f"Raw means: mean affairs without children = {mean0:.3f}, with children = {mean1:.3f} (unadjusted comparison)."
                )
            desc = " ".join(desc_parts)
            return {'object': result, 'description': desc}
        else:
            return {'object': result, 'description': "No usable coefficient or means available in model_output."}

    # If count coefficient present, interpret it
    if result['count_coef'] is not None:
        coef = result['count_coef']
        p = result['count_pvalue']
        irr = result['count_IRR']
        sig = (p is not None and p < 0.05)
        direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
        desc_parts.append(
            f"Model used: {result['model_used']}. In the count equation, the children_yes coefficient = {coef:.4f} "
            f"(p = {p:.4f})" if p is not None else
            f"Model used: {result['model_used']}. In the count equation, the children_yes coefficient = {coef:.4f} (p = NA)"
        )
        if irr is not None:
            desc_parts.append(f"-> incidence-rate ratio (exp(coef)) = {irr:.4f}.")
            if irr < 1:
                desc_parts.append("This implies having children is associated with a lower expected count of reported affairs (multiplicatively).")
            elif irr > 1:
                desc_parts.append("This implies having children is associated with a higher expected count of reported affairs (multiplicatively).")
            else:
                desc_parts.append("This implies no multiplicative change in expected affairs.")
        else:
            if coef < 0:
                desc_parts.append("Negative coefficient (log scale) suggests fewer reported affairs when children are present.")
            elif coef > 0:
                desc_parts.append("Positive coefficient (log scale) suggests more reported affairs when children are present.")
            else:
                desc_parts.append("Coefficient is zero, suggesting no change.")
        desc_parts.append("The effect is " + ("statistically significant at p<0.05." if sig else "not statistically significant at p<0.05."))
        # add raw means for context if available
        if means_by_children:
            m0 = means_by_children.get('mean', {}).get(0)
            m1 = means_by_children.get('mean', {}).get(1)
            if (m0 is not None) and (m1 is not None):
                desc_parts.append(f"(Unadjusted means: without children = {m0:.3f}, with children = {m1:.3f}.)")
        desc = " ".join(desc_parts)
        return {'object': result, 'description': desc}

    # If only inflation coefficient present, interpret its meaning
    if result['infl_coef'] is not None:
        coef = result['infl_coef']
        p = result['infl_pvalue']
        sig = (p is not None and p < 0.05)
        # In zero-inflated models, inflation equation modeled on logit scale: positive coef -> higher log-odds of being an 'excess zero'
        desc_parts.append(
            f"Model used: {result['model_used']}. In the inflation equation, the children_yes coefficient = {coef:.4f} "
            f"(p = {p:.4f})" if p is not None else
            f"Model used: {result['model_used']}. In the inflation equation, the children_yes coefficient = {coef:.4f} (p = NA)"
        )
        if coef > 0:
            desc_parts.append("A positive inflation coefficient means having children is associated with higher odds of being an 'excess zero' (i.e., more likely to report zero affairs).")
        elif coef < 0:
            desc_parts.append("A negative inflation coefficient means having children is associated with lower odds of being an 'excess zero' (i.e., less likely to report zero affairs).")
        else:
            desc_parts.append("Coefficient is zero, implying no association with the excess-zero probability.")
        desc_parts.append("The effect is " + ("statistically significant at p<0.05." if sig else "not statistically significant at p<0.05."))
        if means_by_children:
            m0 = means_by_children.get('mean', {}).get(0)
            m1 = means_by_children.get('mean', {}).get(1)
            if (m0 is not None) and (m1 is not None):
                desc_parts.append(f"(Unadjusted means: without children = {m0:.3f}, with children = {m1:.3f}.)")
        desc = " ".join(desc_parts)
        return {'object': result, 'description': desc}

    # Fallback (should not reach here)
    return {'object': result, 'description': "Could not extract a usable children coefficient from the provided model_output."}