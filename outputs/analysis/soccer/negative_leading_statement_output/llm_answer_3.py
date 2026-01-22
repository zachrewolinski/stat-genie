def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, 95% CIs, and incidence-rate-ratios (IRRs = exp(coef))
    for the key predictors ('SkinToneAvg' for the continuous model and 'DarkBinary'
    for the binary extremes model) from a model_output dict produced by the modeling
    function in the prompt.

    model_output is expected to be a dict with keys:
      - 'model_continuous'
      - 'model_binary_extremes'

    Each value may be:
      - a fitted statsmodels Results-like object (possibly a robust-wrapped results object)
      - or an Exception object (if the previous fitting/robustification step failed)

    The function returns a dict:
      {
        "object": {
          "continuous": <extracted_stats_or_None>,
          "binary": <extracted_stats_or_None>
        },
        "description": <human-readable explanation of what was extracted and the implication>
      }

    Each extracted_stats (when available) is a dict:
      {
        "variable": <name>,
        "coef": float,
        "pvalue": float or None,
        "ci_95": [low, high] or [None, None],
        "irr": float or None,
        "irr_ci_95": [low, high] or [None, None]
      }

    If a model value is an Exception, the function will record that and return None for that model's stats
    along with an explanatory message. This function does not attempt to refit models.
    """
    import numpy as np
    import pandas as pd
    results = {"continuous": None, "binary": None}
    messages = []

    def _extract_from_result(res, varname):
        """
        Try to extract coef, pvalue, conf-int, and compute IRR from a statsmodels-like results object.
        Returns (stats_dict, message)
        """
        if isinstance(res, Exception):
            return None, f"Model unavailable: exception encountered -> {repr(res)}"

        # Check for typical attributes
        if not hasattr(res, "params"):
            return None, "Result object does not have 'params' attribute; cannot extract coefficients."

        try:
            params = res.params
            # params may be a pandas Series or numpy array with index
            if hasattr(params, "index"):
                if varname not in params.index:
                    return None, f"Variable '{varname}' not found in result.params (available: {list(params.index)})"
                coef = float(params[varname])
            else:
                # params is array-like without index - we cannot reliably map variable names
                return None, "Result.params has no index; cannot map variable name to coefficient."

            # p-value extraction
            pval = None
            if hasattr(res, "pvalues"):
                pvals = res.pvalues
                if hasattr(pvals, "index") and varname in pvals.index:
                    pval = float(pvals[varname])

            # confidence interval extraction - res.conf_int() is common, sometimes attribute
            ci_low = ci_high = None
            try:
                if hasattr(res, "conf_int"):
                    # conf_int may be callable or attribute
                    conf = res.conf_int() if callable(res.conf_int) else res.conf_int
                    conf_df = pd.DataFrame(conf, index=params.index)
                    # conf may have two columns (0,1) or named columns
                    if varname in conf_df.index:
                        row = conf_df.loc[varname].values
                        if len(row) >= 2:
                            ci_low, ci_high = float(row[0]), float(row[1])
            except Exception:
                # ignore CI extraction errors
                ci_low = ci_high = None

            irr = float(np.exp(coef)) if coef is not None else None
            irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
            irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

            stats = {
                "variable": varname,
                "coef": coef,
                "pvalue": float(pval) if pval is not None else None,
                "ci_95": [ci_low, ci_high],
                "irr": irr,
                "irr_ci_95": [irr_ci_low, irr_ci_high],
            }
            return stats, "Successfully extracted statistics from results object."
        except Exception as e:
            return None, f"Error while extracting from results object: {repr(e)}"

    # Continuous model: look for 'SkinToneAvg'
    cont_res = model_output.get("model_continuous")
    cont_stats, cont_msg = _extract_from_result(cont_res, "SkinToneAvg")
    results["continuous"] = cont_stats
    messages.append("continuous: " + cont_msg)

    # Binary extremes model: look for 'DarkBinary'
    bin_res = model_output.get("model_binary_extremes")
    bin_stats, bin_msg = _extract_from_result(bin_res, "DarkBinary")
    results["binary"] = bin_stats
    messages.append("binary: " + bin_msg)

    # Build final description:
    if results["continuous"] is None and results["binary"] is None:
        description = (
            "No usable model results were found for either the continuous or binary models. "
            "Extraction failed. Details: " + "; ".join(messages) +
            "  To obtain the final yes/no answer you must re-run the models (or provide results objects) "
            "so that coefficients and standard errors can be read. The original model-fitting step "
            "appears to have raised an AttributeError when attempting to compute cluster-robust SEs."
        )
    else:
        parts = []
        if results["continuous"] is not None:
            s = results["continuous"]
            part = (
                f"Continuous model (SkinToneAvg): coef={s['coef']:.4f}, "
                f"p={s['pvalue']:.3g}" if s['pvalue'] is not None else
                f"Continuous model (SkinToneAvg): coef={s['coef']:.4f}, p=NA"
            )
            if s["irr"] is not None:
                part += f", IRR={s['irr']:.3f}"
            if s["ci_95"][0] is not None and s["ci_95"][1] is not None:
                part += f", 95%CI_coef=[{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}], 95%CI_IRR=[{s['irr_ci_95'][0]:.3f}, {s['irr_ci_95'][1]:.3f}]"
            parts.append(part)
        else:
            parts.append("Continuous model: no extractable results (" + cont_msg + ")")

        if results["binary"] is not None:
            s = results["binary"]
            part = (
                f"Binary extremes model (DarkBinary): coef={s['coef']:.4f}, "
                f"p={s['pvalue']:.3g}" if s['pvalue'] is not None else
                f"Binary extremes model (DarkBinary): coef={s['coef']:.4f}, p=NA"
            )
            if s["irr"] is not None:
                part += f", IRR={s['irr']:.3f}"
            if s["ci_95"][0] is not None and s["ci_95"][1] is not None:
                part += f", 95%CI_coef=[{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}], 95%CI_IRR=[{s['irr_ci_95'][0]:.3f}, {s['irr_ci_95'][1]:.3f}]"
            parts.append(part)
        else:
            parts.append("Binary extremes model: no extractable results (" + bin_msg + ")")

        # Short interpretation guidance
        interpretation = (
            "Interpretation guidance: For each model, if the coefficient is >0 and statistically significant (p < 0.05), "
            "that implies darker skin is associated with a higher rate of red cards (IRR > 1). If coef < 0 and significant, "
            "darker skin is associated with a lower rate (IRR < 1). If not significant, there is no evidence of a difference.\n"
        )
        description = " | ".join(parts) + "   " + interpretation + "Extraction notes: " + "; ".join(messages)

    return {"object": results, "description": description}