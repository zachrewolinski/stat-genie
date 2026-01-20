def extract_final_answer(model_output):
    """
    Extract key statistics about the masfem_z coefficient from the provided model_output.

    Returns a dictionary with:
      - "object": dict with numerical results (coef, se, p-value, 95% CI, IRR and IRR CI,
                  model used, dispersion, supports_hypothesis boolean)
      - "description": short plain-language interpretation of the results in context.
    """
    import numpy as np

    result = {
        "coef": None,
        "se": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "irr": None,
        "irr_ci_lower": None,
        "irr_ci_upper": None,
        "model_used": None,
        "dispersion": None,
        "supports_hypothesis": None,  # True if positive coef and p<0.05
        "notes": None
    }

    # Basic checks
    if not isinstance(model_output, dict):
        return {
            "object": result,
            "description": "model_output is not a dict. Cannot extract results."
        }

    # Prefer Negative Binomial if available and not an error; otherwise use Poisson
    nb_res = model_output.get('nb_results', None)
    poisson_res = model_output.get('poisson_results', None)
    dispersion = model_output.get('dispersion', None)
    result['dispersion'] = float(dispersion) if dispersion is not None else None

    chosen_res = None
    chosen_name = None

    # Helper to detect valid statsmodels results wrapper
    def is_results_wrapper(x):
        # Minimal duck-typing: must have params and pvalues and conf_int methods
        return (hasattr(x, 'params') and hasattr(x, 'pvalues') and hasattr(x, 'conf_int'))

    if nb_res is not None and not (isinstance(nb_res, dict) and 'error' in nb_res) and is_results_wrapper(nb_res):
        chosen_res = nb_res
        chosen_name = 'NegativeBinomial'
    elif poisson_res is not None and is_results_wrapper(poisson_res):
        chosen_res = poisson_res
        chosen_name = 'Poisson'
    else:
        return {
            "object": result,
            "description": "No usable fitted model found in model_output (expected statsmodels results)."
        }

    result['model_used'] = chosen_name

    # Try to extract masfem_z stats
    try:
        params = chosen_res.params
        pvalues = chosen_res.pvalues
        bse = getattr(chosen_res, 'bse', None)

        # Ensure params has an index-like (works with pandas Series)
        param_index = None
        try:
            param_index = list(params.index)
        except Exception:
            # fallback: try to coerce names if params is ndarray
            try:
                param_index = list(getattr(params, 'index', []))
            except Exception:
                param_index = []

        if 'masfem_z' not in param_index:
            result['notes'] = "masfem_z not found in model parameters."
            return {
                "object": result,
                "description": "The model does not contain a parameter named 'masfem_z'."
            }

        coef = float(params['masfem_z'])
        # standard error: prefer bse if available
        se = None
        try:
            if bse is not None:
                se = float(bse['masfem_z'])
        except Exception:
            se = None

        pval = None
        try:
            pval = float(pvalues['masfem_z'])
        except Exception:
            pval = None

        # Confidence interval (95%)
        try:
            ci = chosen_res.conf_int()
            # conf_int() returns array-like; assume index aligns with params
            if hasattr(ci, 'loc'):
                # pandas DataFrame-like
                ci_lower = float(ci.loc['masfem_z', 0])
                ci_upper = float(ci.loc['masfem_z', 1])
            else:
                # fallback: try to find correct row by position
                idx = list(params.index).index('masfem_z')
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
        except Exception:
            ci_lower = None
            ci_upper = None

        # Incidence rate ratio and CI
        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

        # Fill result dict
        result.update({
            "coef": coef,
            "se": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "irr": irr,
            "irr_ci_lower": irr_ci_lower,
            "irr_ci_upper": irr_ci_upper
        })

        # Decision rule: hypothesis predicts that more feminine names -> more fatalities
        # So we consider the hypothesis "supported" if coef > 0 and p < 0.05
        supports = None
        if pval is not None:
            if (coef > 0) and (pval < 0.05):
                supports = True
            elif (pval < 0.05) and (coef < 0):
                supports = False
            else:
                supports = False  # treat non-significant as not supported / inconclusive
        result['supports_hypothesis'] = supports

    except Exception as e:
        result['notes'] = f"Error extracting masfem_z stats: {e}"
        return {
            "object": result,
            "description": "Failed to extract masfem_z coefficient statistics: " + str(e)
        }

    # Construct a concise description
    if result['coef'] is None:
        desc = "Could not extract coefficient for masfem_z."
    else:
        sign = "positive" if result['coef'] > 0 else "negative" if result['coef'] < 0 else "zero"
        pstr = ("p = {:.3g}".format(result['p_value']) if result['p_value'] is not None else "p-value unavailable")
        if result['ci_lower'] is not None and result['ci_upper'] is not None:
            ci_str = "95% CI for coef: [{:.3g}, {:.3g}]".format(result['ci_lower'], result['ci_upper'])
        else:
            ci_str = "CI unavailable"

        if result['irr'] is not None:
            if result['irr_ci_lower'] is not None and result['irr_ci_upper'] is not None:
                irr_str = "IRR = {:.3g} (95% CI: [{:.3g}, {:.3g}])".format(
                    result['irr'], result['irr_ci_lower'], result['irr_ci_upper']
                )
            else:
                irr_str = "IRR = {:.3g}".format(result['irr'])
        else:
            irr_str = "IRR unavailable"

        disp_str = "{:.3g}".format(result['dispersion']) if result['dispersion'] is not None else "N/A"

        if result['supports_hypothesis'] is True:
            concl = "The coefficient on masfem_z is positive and statistically significant, which supports the hypothesis that more-feminine hurricane names are associated with more fatalities."
        elif result['supports_hypothesis'] is False and result['p_value'] is not None and result['p_value'] < 0.05:
            concl = "The coefficient on masfem_z is statistically significant but in the opposite direction (negative), which contradicts the hypothesis."
        else:
            concl = "The association between masfem_z and fatalities is not statistically significant (inconclusive evidence for the hypothesis)."

        desc = (
            f"Model used: {chosen_name}. Coefficient for masfem_z = {result['coef']:.4g} ({sign}); "
            f"{pstr}. {ci_str}. {irr_str}. Dispersion statistic = {disp_str}.\n{concl}"
        )

    return {
        "object": result,
        "description": desc
    }