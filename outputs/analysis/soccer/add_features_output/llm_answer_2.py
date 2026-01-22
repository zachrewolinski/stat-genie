def extract_final_answer(model_output):
    """
    Extracts key statistics for the primary (Dark vs Light) and sensitivity (avgSkin) parameters
    from the provided model_output dictionary produced by the modeling function.

    Returns:
      {
        "object": {
          "primary": { "param_name": str, "coef": float, "se": float, "pvalue": float,
                       "ci_lower": float, "ci_upper": float,
                       "irr": float, "irr_ci_lower": float, "irr_ci_upper": float,
                       "nobs": int (if available)
                     },
          "sensitivity": { same fields for avgSkin param } or None if not found
        },
        "description": str  # brief interpretation and yes/no conclusion about the
                            # question "Are dark-skinned players more likely to receive red cards?"
      }

    The function is written defensively to handle either plain GLM results or results
    returned by get_robustcov_results (cluster-robust wrappers).
    """

    import numpy as np

    def _get_param_info(res, target_substring):
        """
        Search for a parameter name containing target_substring in res.params.index.
        If found, extract coef, robust se, pvalue, conf_int, and compute IRR and IRR CI.
        Returns dict or None if not found.
        """
        try:
            params = res.params
        except Exception:
            return None

        # Identify parameter name (first match containing substring)
        param_name = None
        for name in params.index:
            if target_substring in str(name):
                param_name = name
                break
        if param_name is None:
            return None

        # Extract statistics robustly
        try:
            coef = float(params[param_name])
        except Exception:
            coef = None

        # standard error and pvalue
        try:
            se = float(res.bse[param_name])
        except Exception:
            se = None
        try:
            pvalue = float(res.pvalues[param_name])
        except Exception:
            pvalue = None

        # confidence interval: try DataFrame-like, else fallback by position
        try:
            ci = res.conf_int()
            # If conf_int returns DataFrame-like with index
            if hasattr(ci, "loc"):
                ci_lower = float(ci.loc[param_name, 0])
                ci_upper = float(ci.loc[param_name, 1])
            else:
                # assume numpy array; find index of param
                idx = list(params.index).index(param_name)
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
        except Exception:
            ci_lower = None
            ci_upper = None

        # IRR and IRR CI (exponentiated coefficients)
        try:
            irr = float(np.exp(coef)) if coef is not None else None
        except Exception:
            irr = None
        try:
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            irr_ci_lower = None
            irr_ci_upper = None

        # nobs if available
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None

        return {
            "param_name": str(param_name),
            "coef": coef,
            "se": se,
            "pvalue": pvalue,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "irr": irr,
            "irr_ci_lower": irr_ci_lower,
            "irr_ci_upper": irr_ci_upper,
            "nobs": nobs
        }

    # Validate input
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict as expected."
        }

    # Extract model objects
    primary_key = 'nb_model_dark_vs_light'
    sens_key = 'nb_model_avgSkin'

    primary_res = model_output.get(primary_key)
    sens_res = model_output.get(sens_key)

    # If wrapper objects are present (e.g., results from get_robustcov_results),
    # they still should expose params, bse, pvalues, conf_int, nobs.
    primary_info = None
    sens_info = None

    if primary_res is not None:
        # Look for parameter containing 'DarkPlayer'
        primary_info = _get_param_info(primary_res, 'DarkPlayer')

    if sens_res is not None:
        # avgSkin is expected to be exactly named 'avgSkin'
        sens_info = _get_param_info(sens_res, 'avgSkin')

    # Build conclusion for primary analysis
    conclusion = "Could not find parameters of interest in the provided model output."
    if primary_info is not None:
        # Determine direction and statistical significance (alpha = 0.05)
        coef = primary_info["coef"]
        pval = primary_info["pvalue"]
        irr = primary_info["irr"]

        if (coef is None) or (pval is None):
            conclusion = ("Primary model: parameter '{}' found but coefficient or p-value "
                          "could not be extracted.".format(primary_info["param_name"]))
        else:
            sig = (pval < 0.05)
            if coef > 0 and sig:
                conclusion = ("Primary model: Coefficient for '{}' is positive (coef = {:+.4f}, "
                              "IRR = {:.3f}), p = {:.4g}. This indicates that, controlling for the "
                              "covariates, players coded as '{}' receive red cards at a higher rate "
                              "than the reference category (statistically significant at α=0.05)."
                              ).format(primary_info["param_name"], coef, irr if irr is not None else float("nan"),
                                       pval, primary_info["param_name"])
            elif coef > 0 and (not sig):
                conclusion = ("Primary model: Coefficient for '{}' is positive (coef = {:+.4f}, "
                              "IRR = {:.3f}), but not statistically significant (p = {:.4g}). "
                              "No strong evidence that dark-skinned players receive more red cards."
                              ).format(primary_info["param_name"], coef, irr if irr is not None else float("nan"),
                                       pval)
            elif coef < 0 and sig:
                conclusion = ("Primary model: Coefficient for '{}' is negative (coef = {:+.4f}, "
                              "IRR = {:.3f}), p = {:.4g}. This indicates dark-skinned players receive fewer "
                              "red cards than the reference category (statistically significant at α=0.05)."
                              ).format(primary_info["param_name"], coef, irr if irr is not None else float("nan"),
                                       pval)
            else:
                conclusion = ("Primary model: Coefficient for '{}' is negative (coef = {:+.4f}, "
                              "IRR = {:.3f}), but not statistically significant (p = {:.4g}). "
                              "No strong evidence of a difference in red-card rates by skin-tone category."
                              ).format(primary_info["param_name"], coef, irr if irr is not None else float("nan"),
                                       pval)

    # Add a short summary that includes sensitivity analysis if available
    sens_text = ""
    if sens_info is not None:
        coef = sens_info["coef"]
        pval = sens_info["pvalue"]
        irr = sens_info["irr"]
        if (coef is None) or (pval is None):
            sens_text = " Sensitivity model: avgSkin parameter found but coefficient or p-value missing."
        else:
            if coef > 0 and (pval < 0.05):
                sens_text = (" Sensitivity model: avgSkin coefficient is positive (coef = {:+.4f}, IRR = {:.3f}), "
                             "p = {:.4g} suggesting a dose-response (higher avgSkin -> higher red card rate)."
                             ).format(coef, irr if irr is not None else float("nan"), pval)
            elif coef > 0:
                sens_text = (" Sensitivity model: avgSkin coefficient is positive (coef = {:+.4f}, IRR = {:.3f}), "
                             "but not statistically significant (p = {:.4g}).").format(coef, irr if irr is not None else float("nan"), pval)
            elif coef < 0 and (pval < 0.05):
                sens_text = (" Sensitivity model: avgSkin coefficient is negative (coef = {:+.4f}, IRR = {:.3f}), "
                             "p = {:.4g}, indicating higher avgSkin associated with lower red card rates (statistically significant)."
                             ).format(coef, irr if irr is not None else float("nan"), pval)
            else:
                sens_text = (" Sensitivity model: avgSkin coefficient is negative (coef = {:+.4f}) but not statistically significant (p = {:.4g})."
                             ).format(coef, pval)
    else:
        sens_text = " Sensitivity model: avgSkin parameter not found in the provided results."

    # Build the object to return (numbers + CI)
    returned_object = {
        "primary": primary_info,
        "sensitivity": sens_info
    }

    description = conclusion + sens_text

    return {
        "object": returned_object,
        "description": description
    }