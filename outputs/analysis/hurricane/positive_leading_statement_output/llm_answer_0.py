def extract_final_answer(model_output):
    """
    Extracts key statistics for the name-femininity predictor from the primary
    Negative Binomial model and from robustness models (binary gender, OLS on
    logged damage, MTurk femininity), if present in model_output.

    Returns a dict with:
      - "object": a nested dict of numeric results (coef, se, z, p, 95% CI,
                  incidence-rate-ratio and its CI for count models)
      - "description": a concise interpretation of the primary effect and
                       brief notes about robustness.
    """
    import numpy as np

    results = {
        "primary": None,
        "robust_gender_mf": None,
        "robust_ols_damage": None,
        "robust_mturk": None
    }

    def extract_glm_nb(res, varname):
        """Extract stats for a GLM (NegativeBinomial) results object."""
        try:
            coef = float(res.params[varname])
            se = float(res.bse[varname])
            z = float(coef / se) if se != 0 else None
            p = float(res.pvalues[varname]) if varname in res.pvalues.index else None
            ci_raw = res.conf_int().loc[varname].values
            ci_lower, ci_upper = float(ci_raw[0]), float(ci_raw[1])
            irr = float(np.exp(coef))
            irr_ci_lower, irr_ci_upper = float(np.exp(ci_lower)), float(np.exp(ci_upper))
            return {
                "variable": varname,
                "coef": coef,
                "se": se,
                "z": z,
                "p_value": p,
                "ci_95": [ci_lower, ci_upper],
                "incidence_rate_ratio": irr,
                "irr_95": [irr_ci_lower, irr_ci_upper]
            }
        except Exception as e:
            return {"error": f"Could not extract GLM stats for '{varname}': {e}"}

    def extract_ols(res, varname):
        """Extract stats for an OLS results object."""
        try:
            coef = float(res.params[varname])
            se = float(res.bse[varname])
            t = float(res.tvalues[varname])
            p = float(res.pvalues[varname])
            ci_raw = res.conf_int().loc[varname].values
            ci_lower, ci_upper = float(ci_raw[0]), float(ci_raw[1])
            return {
                "variable": varname,
                "coef": coef,
                "se": se,
                "t": t,
                "p_value": p,
                "ci_95": [ci_lower, ci_upper]
            }
        except Exception as e:
            return {"error": f"Could not extract OLS stats for '{varname}': {e}"}

    # Primary model: nb_model, variable masfem_z
    if model_output is None:
        return {"object": None, "description": "No model_output provided."}

    if "nb_model" in model_output and model_output["nb_model"] is not None:
        nb = model_output["nb_model"]
        if hasattr(nb, "params") and "masfem_z" in nb.params.index:
            results["primary"] = extract_glm_nb(nb, "masfem_z")
        else:
            results["primary"] = {"error": "Primary model present but 'masfem_z' not in parameters."}
    else:
        results["primary"] = {"error": "Primary Negative Binomial model ('nb_model') not present."}

    # Robustness A: binary gender_mf in nb_model_gendermf
    if "nb_model_gendermf" in model_output and model_output["nb_model_gendermf"] is not None:
        nb_g = model_output["nb_model_gendermf"]
        if hasattr(nb_g, "params") and "gender_mf" in nb_g.params.index:
            results["robust_gender_mf"] = extract_glm_nb(nb_g, "gender_mf")
        else:
            results["robust_gender_mf"] = {"error": "Robustness model present but 'gender_mf' not in parameters."}

    # Robustness B: OLS on log_ndam15 (ols_damage)
    if "ols_damage" in model_output and model_output["ols_damage"] is not None:
        ols = model_output["ols_damage"]
        if hasattr(ols, "params") and "masfem_z" in ols.params.index:
            results["robust_ols_damage"] = extract_ols(ols, "masfem_z")
        else:
            results["robust_ols_damage"] = {"error": "OLS model present but 'masfem_z' not in parameters."}

    # Robustness: masfem_mturk_z in nb_model_mturk
    if "nb_model_mturk" in model_output and model_output["nb_model_mturk"] is not None:
        nb_m = model_output["nb_model_mturk"]
        if hasattr(nb_m, "params") and "masfem_mturk_z" in nb_m.params.index:
            results["robust_mturk"] = extract_glm_nb(nb_m, "masfem_mturk_z")
        else:
            results["robust_mturk"] = {"error": "MTurk model present but 'masfem_mturk_z' not in parameters."}

    # Build a brief description/interpretation for the primary result
    desc_lines = []
    prim = results["primary"]
    if prim is None:
        desc_lines.append("Primary result not available.")
    elif "error" in prim:
        desc_lines.append(prim["error"])
    else:
        coef = prim["coef"]
        p = prim["p_value"]
        irr = prim["incidence_rate_ratio"]
        irr_ci = prim["irr_95"]
        sig = ("statistically significant (p < 0.05)" if p is not None and p < 0.05 else
               "not statistically significant (p >= 0.05)" if p is not None else "p-value unavailable")
        desc_lines.append(
            f"Primary NB model: masfem_z coef = {coef:.4f} (SE = {prim['se']:.4f}, z = {prim['z']:.2f}, p = {p:.3g}). "
            f"Interpretation: a 1 SD increase in perceived name femininity multiplies expected fatalities by {irr:.3f} "
            f"(95% CI for IRR: [{irr_ci[0]:.3f}, {irr_ci[1]:.3f}]). This effect is {sig}."
        )

    # Add brief notes about robustness if available
    if results["robust_gender_mf"] is not None:
        r = results["robust_gender_mf"]
        if "error" in r:
            desc_lines.append("Robustness (binary gender): " + r["error"])
        else:
            p = r["p_value"]
            sig = ("statistically significant" if p is not None and p < 0.05 else "not statistically significant")
            desc_lines.append(
                f"Robustness (binary gender): gender_mf IRR = {r['incidence_rate_ratio']:.3f} "
                f"(95% CI [{r['irr_95'][0]:.3f}, {r['irr_95'][1]:.3f}]), {sig} (p = {p:.3g})."
            )

    if results["robust_ols_damage"] is not None:
        r = results["robust_ols_damage"]
        if "error" in r:
            desc_lines.append("Robustness (OLS damage): " + r["error"])
        else:
            p = r["p_value"]
            sig = ("statistically significant" if p is not None and p < 0.05 else "not statistically significant")
            desc_lines.append(
                f"Robustness (OLS on log damage): masfem_z coef = {r['coef']:.4f} (95% CI [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]), {sig} (p = {p:.3g})."
            )

    if results["robust_mturk"] is not None:
        r = results["robust_mturk"]
        if "error" in r:
            desc_lines.append("Robustness (MTurk femininity): " + r["error"])
        else:
            p = r["p_value"]
            sig = ("statistically significant" if p is not None and p < 0.05 else "not statistically significant")
            desc_lines.append(
                f"Robustness (MTurk): masfem_mturk_z IRR = {r['incidence_rate_ratio']:.3f} "
                f"(95% CI [{r['irr_95'][0]:.3f}, {r['irr_95'][1]:.3f}]), {sig} (p = {p:.3g})."
            )

    description = " ".join(desc_lines)

    return {"object": results, "description": description}