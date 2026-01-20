def extract_final_answer(model_output):
    """
    Extracts the effect of 'masfem_center' from the provided model_output dict
    (expected to contain 'ols' and optionally 'neg_binom' results).
    Returns a dict with keys:
      - "object": dict with extracted statistics for OLS and NB (if present) and
                  summary booleans about support for the hypothesis.
      - "description": brief interpretation of the statistics in context.
    """
    import numpy as np

    out = {
        "ols": None,
        "neg_binom": None,
        "support_hypothesis_ols": None,
        "support_hypothesis_neg_binom": None,
        "conclusion": None
    }

    # Helper to safely extract row for variable from results
    def _extract_from_result(res, varname):
        res_dict = {}
        try:
            params = res.params
            if varname not in params.index:
                # sometimes params is a numpy array with positional indexing
                raise KeyError(f"{varname} not found in params")
            coef = float(params[varname])
            # standard error, p-value, t/z-stat
            se = float(res.bse[varname]) if hasattr(res, "bse") else None
            pval = float(res.pvalues[varname]) if hasattr(res, "pvalues") else None
            # conf_int returns array or DataFrame; handle both
            try:
                ci = res.conf_int().loc[varname].tolist()
            except Exception:
                # fallback: conf_int returns ndarray in same order as params
                ci_array = res.conf_int()
                # find index
                try:
                    idx = list(params.index).index(varname)
                    ci = [float(ci_array[idx, 0]), float(ci_array[idx, 1])]
                except Exception:
                    ci = [None, None]
            res_dict.update({
                "coef": coef,
                "se": se,
                "p_value": pval,
                "ci_lower": float(ci[0]) if ci[0] is not None else None,
                "ci_upper": float(ci[1]) if ci[1] is not None else None
            })
        except Exception as e:
            res_dict["error"] = str(e)
        return res_dict

    # Validate input
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output must be a dict containing at least an 'ols' results object."
        }

    var = "masfem_center"

    # Extract from OLS results (expected present)
    ols_res = model_output.get("ols", None)
    if ols_res is None:
        out["ols"] = {"error": "OLS results not found in model_output"}
    else:
        ols_stats = _extract_from_result(ols_res, var)
        out["ols"] = ols_stats
        # Determine support: hypothesis predicts positive coef and statistically significant (alpha=0.05)
        try:
            coef = ols_stats.get("coef", None)
            pval = ols_stats.get("p_value", None)
            support = (coef is not None) and (coef > 0) and (pval is not None) and (pval < 0.05)
            out["support_hypothesis_ols"] = bool(support)
        except Exception:
            out["support_hypothesis_ols"] = None

    # Extract from Negative Binomial robustness if available
    nb_res = model_output.get("neg_binom", None)
    if nb_res is None:
        # If NB failed, there may be an error string in model_output
        nb_error = model_output.get("neg_binom_error", None)
        out["neg_binom"] = {"error": nb_error} if nb_error else {"note": "No negative binomial model present"}
        out["support_hypothesis_neg_binom"] = None
    else:
        nb_stats = _extract_from_result(nb_res, var)
        # For count model, also compute IRR = exp(coef) and CI for IRR
        try:
            coef = nb_stats.get("coef", None)
            if coef is not None:
                irr = float(np.exp(coef))
                ci_low = nb_stats.get("ci_lower", None)
                ci_high = nb_stats.get("ci_upper", None)
                irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
                nb_stats.update({
                    "irr": irr,
                    "irr_ci_lower": irr_ci_low,
                    "irr_ci_upper": irr_ci_high,
                    # percent change approximation
                    "irr_percent_change": (irr - 1.0) * 100.0
                })
        except Exception:
            pass
        out["neg_binom"] = nb_stats
        try:
            coef = nb_stats.get("coef", None)
            pval = nb_stats.get("p_value", None)
            support_nb = (coef is not None) and (coef > 0) and (pval is not None) and (pval < 0.05)
            out["support_hypothesis_neg_binom"] = bool(support_nb)
        except Exception:
            out["support_hypothesis_neg_binom"] = None

    # Synthesize a short conclusion
    try:
        ols_sup = out["support_hypothesis_ols"]
        nb_sup = out["support_hypothesis_neg_binom"]
        if ols_sup is True and (nb_sup is True or nb_sup is None):
            concl = "Results support the hypothesis: more feminine names are associated with higher fatalities (OLS significant; NB supports if present)."
        elif ols_sup is True and nb_sup is False:
            concl = "OLS shows a significant positive association consistent with the hypothesis, but the negative binomial robustness does not support it."
        elif ols_sup is False and nb_sup is True:
            concl = "Negative binomial robustness shows a significant positive association consistent with the hypothesis, but OLS does not."
        elif ols_sup is False and nb_sup is False:
            concl = "Neither OLS nor negative binomial show a statistically significant positive association; results do not support the hypothesis."
        else:
            concl = "Mixed or incomplete evidence: check extracted statistics in 'object' for details."
        out["conclusion"] = concl
    except Exception:
        out["conclusion"] = "Could not form conclusion from results."

    # Prepare final return structure
    return {
        "object": out,
        "description": (
            "Extracted statistics for the effect of masfem_center (higher = more feminine name) on fatalities. "
            "For OLS (DV = log(1 + alldeaths)): 'coef' is the change in log-deaths per one-unit increase in masfem_center; "
            "positive coef means more feminine names are associated with higher fatalities (consistent with the hypothesis). "
            "For Negative Binomial (DV = raw alldeaths): 'coef' is on the log count scale; 'irr' = exp(coef) is the multiplicative "
            "change in expected deaths per one-unit increase in masfem_center (e.g., irr=1.10 means ~10% higher expected deaths). "
            "P-values and 95% confidence intervals are provided. The 'support_hypothesis_...' booleans indicate whether each model's "
            "estimate is positive and statistically significant at alpha=0.05. The 'conclusion' summarizes whether the evidence supports "
            "the hypothesis."
        )
    }