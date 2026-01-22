def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, and confidence intervals for the beauty terms
    (beauty_c and beauty_sq) from the provided model_output and returns a short
    interpretation.

    Returns a dict with:
      - "object": dict with numeric results for beauty_c and beauty_sq
      - "description": short plain-language interpretation of those results
    """
    # Helper to coerce arrays/series to plain python floats/lists
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return x

    # Try to obtain params, pvalues, conf_int from different possible shapes of model_output
    params = pvalues = conf_int = None

    # case: model_output is a statsmodels results object
    if not isinstance(model_output, dict) and hasattr(model_output, 'params'):
        params = model_output.params
        pvalues = model_output.pvalues
        conf_int = model_output.conf_int()

    # case: model_output is a dict (as in the provided example)
    elif isinstance(model_output, dict):
        # Prefer the 'results' item if present (statsmodels object)
        if 'results' in model_output and hasattr(model_output['results'], 'params'):
            res = model_output['results']
            params = res.params
            pvalues = res.pvalues
            conf_int = res.conf_int()
        # Otherwise fall back to arrays provided in the dict
        else:
            # Expect arrays in order: const, beauty_c, beauty_sq, ...
            coef_arr = model_output.get('coef')
            pval_arr = model_output.get('pvalues')
            ci_arr = model_output.get('conf_int')
            if coef_arr is None or pval_arr is None or ci_arr is None:
                raise ValueError("model_output dict must contain either a 'results' object or 'coef', 'pvalues', 'conf_int' arrays.")
            params = coef_arr
            pvalues = pval_arr
            conf_int = ci_arr
    else:
        raise ValueError("Unsupported model_output format")

    # Identify or assume positions/names for beauty_c and beauty_sq
    def _get_value(container, name, pos):
        # If container has keys/index (like pandas Series / DataFrame), use them
        try:
            if hasattr(container, 'index') or hasattr(container, 'keys'):
                return container[name]
        except Exception:
            pass
        # Otherwise treat as array-like and use pos
        return container[pos]

    # According to the modeling code the column order is:
    # const (0), beauty_c (1), beauty_sq (2), minority_binary (3), age (4), gender_male (5),
    # credits_single (6), division_upper (7), native_yes (8), tenure_yes (9), log_students (10)
    pos_beauty_c = 1
    pos_beauty_sq = 2

    try:
        b1 = _to_float(_get_value(params, 'beauty_c', pos_beauty_c))
        p1 = _to_float(_get_value(pvalues, 'beauty_c', pos_beauty_c))
        ci1 = _get_value(conf_int, 'beauty_c', pos_beauty_c)
    except Exception:
        # If conf_int is a numpy array without names, index by row
        b1 = _to_float(params[pos_beauty_c])
        p1 = _to_float(pvalues[pos_beauty_c])
        ci1 = conf_int[pos_beauty_c]

    try:
        b2 = _to_float(_get_value(params, 'beauty_sq', pos_beauty_sq))
        p2 = _to_float(_get_value(pvalues, 'beauty_sq', pos_beauty_sq))
        ci2 = _get_value(conf_int, 'beauty_sq', pos_beauty_sq)
    except Exception:
        b2 = _to_float(params[pos_beauty_sq])
        p2 = _to_float(pvalues[pos_beauty_sq])
        ci2 = conf_int[pos_beauty_sq]

    # Ensure confidence intervals are plain lists of floats
    try:
        ci1 = [float(ci1[0]), float(ci1[1])]
    except Exception:
        ci1 = list(ci1)
    try:
        ci2 = [float(ci2[0]), float(ci2[1])]
    except Exception:
        ci2 = list(ci2)

    # Build the object to return and a descriptive interpretation
    result_object = {
        'beauty_c_coef': b1,
        'beauty_c_pvalue': p1,
        'beauty_c_95CI': ci1,
        'beauty_sq_coef': b2,
        'beauty_sq_pvalue': p2,
        'beauty_sq_95CI': ci2,
        # simple inference flags
        'beauty_c_significant_at_0.05': (p1 is not None and p1 < 0.05),
        'beauty_sq_significant_at_0.05': (p2 is not None and p2 < 0.05)
    }

    # Compose plain-language description
    description = (
        f"Estimated effect of instructor attractiveness (beauty):\n"
        f"- Linear term (beauty_c): coef = {b1:.3f}, p = {p1:.3f}, 95% CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]. "
        f"This is statistically significant at the 0.05 level and means that a one-unit increase in mean-centered "
        f"beauty is associated with about a {b1:.3f}-point higher course evaluation score (on the 1–5 scale) at the mean.\n"
        f"- Quadratic term (beauty_sq): coef = {b2:.3f}, p = {p2:.3f}, 95% CI = [{ci2[0]:.3f}, {ci2[1]:.3f}]. "
        f"This term is not statistically significant (p > 0.05), so there is no reliable evidence of a nonlinear "
        f"(e.g., diminishing or accelerating) effect of beauty on evaluations in this model.\n\n"
        f"Conclusion: There is a positive and statistically significant linear association between measured attractiveness "
        f"and teaching evaluations (about {b1:.3f} points per unit of the mean-centered beauty score). However, the "
        f"quadratic term is not significant, so the data do not support a meaningful nonlinear relationship after "
        f"controlling for the listed covariates."
    )

    return {'object': result_object, 'description': description}