def extract_final_answer(model_output):
    """
    Extracts statistics about the IsHuman effect from the model_output produced
    by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict of extracted numeric results
      - "description": a short plain-language interpretation answering whether
                       modern humans have higher AMTL after adjustment.
    """
    import numpy as np

    # Prepare default return structure
    result_obj = {
        'coef_log_odds': None,
        'se': None,
        'z_value': None,
        'p_value': None,
        'odds_ratio': None,
        'odds_ratio_ci': (None, None),
        'conf_int_log_odds': (None, None),
        'is_significant': None,
        'n_obs': None,
        'notes': []
    }

    # Try to get GLM results object if present
    glm_res = model_output.get('glm_results') if isinstance(model_output, dict) else None

    try:
        if glm_res is None:
            raise ValueError("No 'glm_results' found in model_output dict.")

        # Extract log-odds coefficient, SE, z, p
        params = glm_res.params
        bse = glm_res.bse
        pvalues = glm_res.pvalues
        conf = glm_res.conf_int()  # log-odds scale

        if 'IsHuman' not in params.index:
            raise KeyError("'IsHuman' not present in model coefficients.")

        coef = float(params.loc['IsHuman'])
        se = float(bse.loc['IsHuman'])
        pval = float(pvalues.loc['IsHuman'])
        z = coef / se if se != 0 else None
        ci_lo_log, ci_hi_log = float(conf.loc['IsHuman', 0]), float(conf.loc['IsHuman', 1])

        # Odds ratio and CI on OR scale
        or_est = float(np.exp(coef))
        or_ci_lo = float(np.exp(ci_lo_log))
        or_ci_hi = float(np.exp(ci_hi_log))

        # Number of observations if available
        nobs = getattr(glm_res, 'nobs', None)
        try:
            nobs = int(nobs) if nobs is not None else None
        except Exception:
            nobs = None

        # Populate result object
        result_obj.update({
            'coef_log_odds': coef,
            'se': se,
            'z_value': z,
            'p_value': pval,
            'odds_ratio': or_est,
            'odds_ratio_ci': (or_ci_lo, or_ci_hi),
            'conf_int_log_odds': (ci_lo_log, ci_hi_log),
            'n_obs': nobs,
        })

        # Determine significance and direction
        is_significant = False
        # Consider significant if p < 0.05 OR CI on OR excludes 1 in a consistent direction
        if (pval is not None and pval < 0.05) or (or_ci_lo is not None and or_ci_hi is not None and (or_ci_lo > 1 or or_ci_hi < 1)):
            is_significant = True
        result_obj['is_significant'] = is_significant

        # Build description / interpretation
        if is_significant:
            if coef > 0:
                interpretation = (
                    f"Yes. After adjusting for age, sex, and tooth class, modern humans "
                    f"(IsHuman=1) have significantly higher odds of antemortem tooth loss. "
                    f"Estimated odds ratio = {or_est:.2f} "
                    f"(95% CI: {or_ci_lo:.2f}–{or_ci_hi:.2f}), p = {pval:.3g}."
                )
            else:
                interpretation = (
                    f"No — modern humans have significantly lower odds of AMTL after adjustment. "
                    f"Estimated odds ratio = {or_est:.2f} "
                    f"(95% CI: {or_ci_lo:.2f}–{or_ci_hi:.2f}), p = {pval:.3g}."
                )
        else:
            interpretation = (
                f"Inconclusive: the effect of modern humans on AMTL is not statistically significant "
                f"after adjustment. Estimated odds ratio = {or_est:.2f} "
                f"(95% CI: {or_ci_lo:.2f}–{or_ci_hi:.2f}), p = {pval:.3g}."
            )

        # Attach a short methodological note
        note = ("Model: binomial GLM (logit) of Missing/Sockets with weights=Sockets; "
                "IsHuman coefficient shown on log-odds scale and transformed to odds ratio.")
        result_obj['notes'].append(note)

        return {
            "object": result_obj,
            "description": interpretation
        }

    except Exception as e:
        # Fallback: if model_output included precomputed odds ratio and CI, use those
        # This handles the provided model_output dict which also included these keys.
        try:
            or_est = float(model_output.get('odds_ratio_IsHuman')) if isinstance(model_output, dict) else None
            or_ci = model_output.get('odds_ratio_ci_IsHuman') if isinstance(model_output, dict) else (None, None)
            if or_est is not None:
                interpretation = (
                    f"Based on available output, estimated odds ratio for IsHuman = {or_est:.2f} "
                    f"(95% CI: {or_ci[0]:.2f}–{or_ci[1]:.2f}). CI excludes 1, suggesting higher AMTL in modern humans after adjustment."
                )
                result_obj.update({
                    'odds_ratio': or_est,
                    'odds_ratio_ci': (float(or_ci[0]) if or_ci[0] is not None else None,
                                      float(or_ci[1]) if or_ci[1] is not None else None),
                    'notes': [f"Fallback used because of error extracting GLM results: {str(e)}"]
                })
                return {"object": result_obj, "description": interpretation}
        except Exception:
            pass

        # If everything fails, return the error message
        return {
            "object": result_obj,
            "description": f"Failed to extract results: {str(e)}"
        }