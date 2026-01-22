def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, and 95% CIs for the femininity / gender predictors
    from the provided model_output dictionary and returns a concise conclusion.

    Returns a dict with:
      - "object": dictionary of extracted numeric results (primary model + sensitivities)
      - "description": plain-language interpretation about whether more feminine names
                       are associated with higher fatalities.
    """
    import math
    import numpy as np

    def _extract_from_model(model, varname):
        """Helper to safely extract coef, pval, ci for varname from a statsmodels result."""
        if model is None:
            return None
        try:
            params = model.params
            pvalues = model.pvalues
            conf = model.conf_int()
            # get coef
            coef = float(params[varname])
            pval = float(pvalues[varname])
            # conf may be DataFrame or ndarray; try label-based then position-based
            try:
                ci_lower, ci_upper = conf.loc[varname].tolist()
            except Exception:
                # fallback: find index position of varname in params
                idx = list(params.index).index(varname)
                ci_lower, ci_upper = conf[idx].tolist()
            return {"coef": coef, "pvalue": pval, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}
        except Exception as e:
            # If extraction fails, return the exception message for debugging
            return {"error": f"extraction_failed: {e}"}

    results = {}

    # Primary continuous femininity (OLS on log_alldeaths)
    ols_masfem = model_output.get("ols_masfem")
    primary = _extract_from_model(ols_masfem, "masfem_z")
    results["ols_masfem"] = primary

    # Binary gender (OLS sensitivity)
    ols_gender = model_output.get("ols_gender")
    gender = _extract_from_model(ols_gender, "gender_mf_num")
    results["ols_gender"] = gender

    # Negative Binomial (count) sensitivity
    nb = model_output.get("nb_masfem")
    nb_res = _extract_from_model(nb, "masfem_z")
    # if available, compute exponentiated effect (multiplicative change) and CI
    if isinstance(nb_res, dict) and nb_res is not None and "coef" in nb_res:
        try:
            nb_res["exp_coef"] = float(np.exp(nb_res["coef"]))
            nb_res["exp_ci_lower"] = float(np.exp(nb_res["ci_lower"]))
            nb_res["exp_ci_upper"] = float(np.exp(nb_res["ci_upper"]))
        except Exception:
            pass
    results["nb_masfem"] = nb_res

    # MTurk continuous femininity (sensitivity OLS)
    ols_mturk = model_output.get("ols_mturk")
    mturk = _extract_from_model(ols_mturk, "masfem_mturk_z")
    results["ols_mturk"] = mturk

    # Determine overall evidence: any model with coef>0 and p<0.05 (supports hypothesis),
    # or coef<0 and p<0.05 (opposes hypothesis). If none significant, conclude no evidence.
    evidence = {"supports_hypothesis": False, "opposes_hypothesis": False, "significant_models": []}
    for name, info in results.items():
        if not info or "error" in info:
            continue
        if ("coef" in info) and (isinstance(info["coef"], (int, float))):
            coef = info["coef"]
            p = info.get("pvalue", math.nan)
            if (p is not None) and (p < 0.05):
                if coef > 0:
                    evidence["supports_hypothesis"] = True
                    evidence["significant_models"].append(name)
                elif coef < 0:
                    evidence["opposes_hypothesis"] = True
                    evidence["significant_models"].append(name)

    # Build plain-language description using extracted numbers (prefer primary model)
    desc_lines = []
    if primary and "coef" in primary:
        coef = primary["coef"]
        p = primary["pvalue"]
        ci_l = primary["ci_lower"]
        ci_u = primary["ci_upper"]
        desc_lines.append(
            f"Primary OLS (log fatalities ~ masfem_z): coef = {coef:.4f}, 95% CI [{ci_l:.3f}, {ci_u:.3f}], p = {p:.3f}."
        )
        desc_lines.append(
            "Interpretation: This is the expected change in log1p(fatalities) per 1 SD increase in perceived femininity."
        )
    else:
        desc_lines.append("Primary OLS (masfem_z) results not available.")

    # Summarize sensitivities
    if gender and "coef" in gender:
        desc_lines.append(
            f"Sensitivity OLS (binary gender): coef = {gender['coef']:.4f}, 95% CI [{gender['ci_lower']:.3f}, {gender['ci_upper']:.3f}], p = {gender['pvalue']:.3f}."
        )
    if mturk and "coef" in mturk:
        desc_lines.append(
            f"Sensitivity OLS (MTurk masfem): coef = {mturk['coef']:.4f}, 95% CI [{mturk['ci_lower']:.3f}, {mturk['ci_upper']:.3f}], p = {mturk['pvalue']:.3f}."
        )
    if nb_res and "coef" in nb_res:
        desc_lines.append(
            f"Negative Binomial (counts): log-coef = {nb_res['coef']:.4f}, 95% CI [{nb_res['ci_lower']:.3f}, {nb_res['ci_upper']:.3f}], p = {nb_res['pvalue']:.3f}."
        )
        if "exp_coef" in nb_res:
            desc_lines.append(
                f"  => multiplicative effect exp(coef) = {nb_res['exp_coef']:.3f}, 95% CI [{nb_res['exp_ci_lower']:.3f}, {nb_res['exp_ci_upper']:.3f}]."
            )

    # Final conclusion based on significance across models
    if evidence["supports_hypothesis"] and not evidence["opposes_hypothesis"]:
        final_statement = (
            "Conclusion: At least one model shows a statistically significant positive association "
            "between femininity and fatalities (supports the hypothesis)."
        )
    elif evidence["opposes_hypothesis"] and not evidence["supports_hypothesis"]:
        final_statement = (
            "Conclusion: At least one model shows a statistically significant negative association "
            "(opposes the hypothesis)."
        )
    else:
        final_statement = (
            "Conclusion: No consistent evidence that more feminine hurricane names are associated with higher fatalities. "
            "All reported femininity / gender coefficients are small and not statistically significant (p >= 0.05)."
        )

    desc_lines.append(final_statement)

    description = " ".join(desc_lines)

    # Package the object to return: extracted numeric results plus a simple boolean verdict
    final_object = {
        "extracted_results": results,
        "evidence_summary": evidence,
        "verdict": {
            # True if at least one model supports the hypothesis (coef>0 & p<0.05)
            "supports_hypothesis": bool(evidence["supports_hypothesis"]),
            # True if at least one model opposes (coef<0 & p<0.05)
            "opposes_hypothesis": bool(evidence["opposes_hypothesis"]),
            # Overall: require at least one supporting significant model and no opposing significant models
            "overall_support": bool(evidence["supports_hypothesis"] and not evidence["opposes_hypothesis"])
        }
    }

    return {"object": final_object, "description": description}