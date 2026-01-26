def extract_final_answer(model_output):
    """
    Extracts statistics for the masfem_z coefficient from the provided model_output.

    Returns a dictionary with:
      - "object": a dict with numeric results from the main Negative Binomial GLM
                  (and the robustness OLS on log damage if available)
      - "description": a brief textual interpretation of the results in context.

    Expected input format:
      model_output = {
        'main_nb_glm': <statsmodels GLMResultsWrapper>,
        'robustness': {'ols_log_damage': <statsmodels RegressionResultsWrapper> or None}
      }
    """
    import numpy as np

    out = {"object": {}, "description": ""}

    # Helper to format extracted stats
    def _extract_from_results(res, param_name, model_label, is_glm=True):
        stats = {"model": model_label}
        try:
            params = res.params
            if param_name not in params.index:
                stats["error"] = f"Parameter '{param_name}' not found in model."
                return stats
            coef = float(params[param_name])
        except Exception as e:
            stats["error"] = f"Could not extract params: {e}"
            return stats

        # Standard error, test stat, pvalue
        try:
            se = float(res.bse[param_name])
        except Exception:
            se = None
        try:
            if is_glm:
                # GLM: z-value
                test_stat = float(coef / se) if (se is not None and se != 0) else None
                stat_name = "z"
            else:
                # OLS: t-value
                test_stat = float(res.tvalues[param_name]) if param_name in res.tvalues.index else (float(coef / se) if (se is not None and se != 0) else None)
                stat_name = "t"
        except Exception:
            test_stat = None
            stat_name = "stat"

        try:
            pval = float(res.pvalues[param_name])
        except Exception:
            pval = None

        # Confidence intervals
        try:
            ci = res.conf_int().loc[param_name].astype(float)
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        except Exception:
            ci_lower = None
            ci_upper = None

        stats.update({
            "coef": coef,
            "se": se,
            stat_name: test_stat,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

        # If GLM with log link, also provide incidence rate ratio (IRR) and its CI
        try:
            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
            stats.update({
                "irr": irr,
                "irr_ci_lower": irr_ci_lower,
                "irr_ci_upper": irr_ci_upper,
                "irr_interpretation": (
                    "Multiplicative factor for expected fatalities per 1 SD increase in femininity"
                )
            })
        except Exception:
            # If exponentiation fails, skip IRR
            pass

        return stats

    # Extract from main negative binomial GLM
    try:
        main = model_output.get("main_nb_glm", None)
    except Exception:
        main = None

    if main is None:
        out["object"]["main_nb_glm"] = {"error": "main_nb_glm not found in model_output"}
    else:
        nb_stats = _extract_from_results(main, "masfem_z", model_label="NegativeBinomial(GLM)", is_glm=True)
        out["object"]["main_nb_glm"] = nb_stats

    # Extract from robustness OLS if present
    robustness = model_output.get("robustness", {})
    ols_mod = robustness.get("ols_log_damage", None) if isinstance(robustness, dict) else None
    if ols_mod is None:
        out["object"]["ols_log_damage"] = {"note": "No OLS robustness model available"}
    else:
        ols_stats = _extract_from_results(ols_mod, "masfem_z", model_label="OLS(log_damage + 1)", is_glm=False)
        out["object"]["ols_log_damage"] = ols_stats

    # Build a short description/interpretation focusing on the hypothesis:
    # "More feminine hurricane names lead to fewer fatalities"
    desc_lines = []
    nb = out["object"].get("main_nb_glm", {})
    if "error" in nb:
        desc_lines.append("Negative binomial model results could not be extracted: " + nb.get("error"))
    else:
        if "coef" in nb and nb.get("coef") is not None:
            coef = nb["coef"]
            p = nb.get("p_value")
            irr = nb.get("irr")
            # Direction
            if coef < 0:
                direction = "associated with fewer expected fatalities"
            elif coef > 0:
                direction = "associated with more expected fatalities"
            else:
                direction = "no apparent association with expected fatalities"

            # Significance statement
            if p is not None:
                sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
            else:
                sig = "p-value unavailable"

            desc_lines.append(
                f"In the Negative Binomial GLM, a 1 SD increase in perceived femininity (masfem_z) has "
                f"a coefficient = {coef:.4g}"
                + (f", p = {p:.3g}" if p is not None else "")
                + f". This is {direction}; the estimated incidence rate ratio (exp(coef)) = {irr:.4g}" if irr is not None else ""
                + f". The effect is {sig}."
            )
            # Add CI if present
            if nb.get("ci_lower") is not None and nb.get("ci_upper") is not None:
                desc_lines.append(
                    f"95% CI for coef: [{nb['ci_lower']:.4g}, {nb['ci_upper']:.4g}]; "
                    f"95% CI for IRR: [{nb.get('irr_ci_lower', float('nan')):.4g}, {nb.get('irr_ci_upper', float('nan')):.4g}]."
                )
        else:
            desc_lines.append("masfem_z coefficient not available in negative binomial model output.")

    # Include a brief note about robustness OLS
    ols = out["object"].get("ols_log_damage", {})
    if "note" in ols:
        desc_lines.append("No OLS(log damage) robustness model was available.")
    elif "error" in ols:
        desc_lines.append("OLS robustness model could not be extracted: " + ols.get("error"))
    else:
        if "coef" in ols and ols.get("coef") is not None:
            coef_o = ols["coef"]
            p_o = ols.get("p_value")
            if coef_o < 0:
                dir_o = "negative (fewer logged damages with more feminine names)"
            elif coef_o > 0:
                dir_o = "positive (more logged damages with more feminine names)"
            else:
                dir_o = "no apparent association"
            if p_o is not None:
                sig_o = "statistically significant (p < 0.05)" if p_o < 0.05 else "not statistically significant (p >= 0.05)"
            else:
                sig_o = "p-value unavailable"
            desc_lines.append(
                f"Robustness OLS on log-damage: coef = {coef_o:.4g}"
                + (f", p = {p_o:.3g}" if p_o is not None else "")
                + f"; direction: {dir_o}; {sig_o}."
            )
        else:
            desc_lines.append("masfem_z coefficient not available in OLS robustness model output.")

    # Final concise verdict hint (do not over-claim)
    # We will indicate whether the main model provides evidence consistent with the hypothesis
    if "coef" in nb and nb.get("coef") is not None and nb.get("p_value") is not None:
        if nb["coef"] < 0 and nb["p_value"] < 0.05:
            conclusion = "The main model's estimate is consistent with the hypothesis (more feminine names → fewer fatalities) and is statistically significant."
        elif nb["coef"] < 0 and nb["p_value"] >= 0.05:
            conclusion = "The main model's estimate is in the hypothesized direction (more feminine names → fewer fatalities) but is not statistically significant."
        elif nb["coef"] > 0 and nb["p_value"] < 0.05:
            conclusion = "The main model's estimate is in the opposite direction to the hypothesis and is statistically significant."
        else:
            conclusion = "The main model does not provide statistically significant evidence supporting the hypothesis."
        desc_lines.append(conclusion)

    out["description"] = " ".join(desc_lines)
    return out