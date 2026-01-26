def extract_final_answer(model_output):
    """
    Extracts statistics about the female coefficient from the model output and
    returns a concise object and human-readable interpretation.

    Returns a dictionary with keys:
      - "object": dict with numeric results {coef_log_odds, robust_se, pvalue, ame_prob_diff, significant}
      - "description": brief explanation of the extracted numbers in plain language
    """
    import math

    # Helpers to safely extract values from possible containers
    def _safe_get(container, key):
        if container is None:
            return None
        # dict-like
        try:
            return container[key]
        except Exception:
            pass
        # namespace/object with attributes
        try:
            return getattr(container, key)
        except Exception:
            pass
        # pandas Series / Indexable by label
        try:
            return container.loc[key]
        except Exception:
            pass
        # fallback: try accessing as item 'female' in keys
        return None

    # Try direct entries first (these were produced by the modeling function)
    coef = None
    robust_se = None
    pvalue = None
    ame = None

    # If top-level keys exist, prefer them
    if isinstance(model_output, dict):
        ame = model_output.get('ame_female', None)
        pvalue = model_output.get('female_pvalue', None)
        # robust_results likely present
        robust_res = model_output.get('robust_results', None)
    else:
        robust_res = model_output

    # Extract from robust_results if present
    if robust_res is not None:
        # robust_res may be a SimpleNamespace with .params, .pvalues, .bse
        coef_val = _safe_get(robust_res, 'params')
        if coef_val is not None:
            # try to get 'female' entry from params
            try:
                coef = float(coef_val['female'])
            except Exception:
                try:
                    coef = float(coef_val.loc['female'])
                except Exception:
                    coef = None

        se_val = _safe_get(robust_res, 'bse')
        if se_val is not None:
            try:
                robust_se = float(se_val['female'])
            except Exception:
                try:
                    robust_se = float(se_val.loc['female'])
                except Exception:
                    robust_se = None

        pval_val = _safe_get(robust_res, 'pvalues')
        if pval_val is not None and pvalue is None:
            try:
                pvalue = float(pval_val['female'])
            except Exception:
                try:
                    pvalue = float(pval_val.loc['female'])
                except Exception:
                    pvalue = None

    # If ame or pvalue still missing, try top-level keys again
    if ame is None and isinstance(model_output, dict):
        ame = model_output.get('ame_female', None)
    if pvalue is None and isinstance(model_output, dict):
        pvalue = model_output.get('female_pvalue', None)

    # Coerce numeric types if possible
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    coef = _to_float(coef)
    robust_se = _to_float(robust_se)
    pvalue = _to_float(pvalue)
    ame = _to_float(ame)

    # Determine statistical significance (conservative: require pvalue available)
    significant = None
    if pvalue is not None:
        significant = bool(pvalue < 0.05)

    # Prepare the returned numeric object
    numeric_object = {
        'coef_log_odds': coef,
        'robust_se': robust_se,
        'pvalue': pvalue,
        # AME is returned as probability difference (e.g., 0.0264 = +2.64 percentage points)
        'ame_prob_diff': ame,
        'significant_at_0.05': significant
    }

    # Build human-readable description
    if coef is None and ame is None:
        description = "Could not extract female coefficient or AME from the provided model output."
    else:
        parts = []
        if coef is not None:
            parts.append(f"The estimated female coefficient on the log-odds scale is {coef:.4f}.")
            if robust_se is not None:
                parts.append(f"Robust SE = {robust_se:.4f}.")
            if pvalue is not None:
                parts.append(f"Robust p-value = {pvalue:.2e}.")
                if significant:
                    parts.append("This is statistically significant at the 5% level.")
                else:
                    parts.append("This is not statistically significant at the 5% level.")
        if ame is not None:
            parts.append(f"The average marginal effect (probability change when switching female 0→1 at covariate means) is {ame:.4f}, i.e. about {ame*100:.2f} percentage points.")
        # short substantive interpretation
        if coef is not None:
            direction = "increase" if coef > 0 else "decrease"
            parts.append(f"In substantive terms, being female is associated with a {direction} in the probability of mortgage approval (positive coef implies higher approval probability).")
        description = " ".join(parts)

    return {
        "object": numeric_object,
        "description": description
    }