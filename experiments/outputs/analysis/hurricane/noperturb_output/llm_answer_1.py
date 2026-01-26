def extract_final_answer(model_output):
    """
    Extracts coefficients, confidence intervals, p-values, and interpretable effect
    sizes for the focal independent variables (masfem_z and gender_mf) from the
    provided model_output dict.

    Returns:
      {
        "object": {
           "deaths": { <variable>: {coef, se, pvalue, 95ci, irr (for NB), irr_95ci, supports_hypothesis}, ...},
           "damages": { <variable>: {coef, se, pvalue, 95ci, supports_hypothesis}, ...},
           "alpha": <optional overdispersion/nb params if available>
        },
        "description": "<brief human-readable interpretation>"
      }
    """
    import math
    import numpy as np

    out = {"deaths": {}, "damages": {}}
    description_lines = []
    sig_alpha = 0.05

    # Helper to safely extract stats for a variable from a results object
    def extract_stats(res, varname):
        if res is None:
            return None
        try:
            params = res.params
        except Exception:
            return None
        if varname not in params.index:
            return None
        coef = float(params.loc[varname])
        # Std err and pvalue extraction may fail on some wrappers; guard with getattr
        try:
            se = float(res.bse.loc[varname])
        except Exception:
            se = None
        try:
            pvalue = float(res.pvalues.loc[varname])
        except Exception:
            pvalue = None
        try:
            ci_df = res.conf_int(alpha=0.05)
            ci_lower = float(ci_df.loc[varname, 0])
            ci_upper = float(ci_df.loc[varname, 1])
            ci = (ci_lower, ci_upper)
        except Exception:
            ci = (None, None)

        return {"coef": coef, "se": se, "pvalue": pvalue, "95ci": ci}

    # Process deaths model (Negative Binomial GLM)
    nb_res = model_output.get("deaths_model")
    for var in ["masfem_z", "gender_mf"]:
        stats = extract_stats(nb_res, var)
        if stats is None:
            out["deaths"][var] = None
            continue
        # Compute incidence rate ratio (IRR) and its 95% CI by exponentiating
        irr = math.exp(stats["coef"])
        ci = stats["95ci"]
        irr_ci = (math.exp(ci[0]) if ci[0] is not None else None, math.exp(ci[1]) if ci[1] is not None else None)
        stats["irr"] = irr
        stats["irr_95ci"] = irr_ci
        # Decide whether this result supports the hypothesis:
        # Hypothesis: more-feminine names -> fewer precautions -> MORE deaths/damages.
        # So a positive coef / IRR > 1 supports the hypothesis.
        supports = None
        if stats["pvalue"] is not None:
            supports = (stats["coef"] > 0) and (stats["pvalue"] < sig_alpha)
        else:
            supports = (stats["coef"] > 0)
        stats["supports_hypothesis"] = supports
        out["deaths"][var] = stats

    # Capture NB alpha/scale if available (some GLM NB results expose 'scale' or 'deviance' or model.family)
    try:
        # In statsmodels GLMResultsWrapper with NegativeBinomial family, there may be a 'scale' attribute.
        if nb_res is not None:
            alpha = getattr(nb_res, "scale", None)
        else:
            alpha = None
    except Exception:
        alpha = None
    if alpha is not None:
        out["alpha"] = alpha

    # Process damages model (OLS on log_ndam15)
    ols_res = model_output.get("damage_model")
    for var in ["masfem_z", "gender_mf"]:
        stats = extract_stats(ols_res, var)
        if stats is None:
            out["damages"][var] = None
            continue
        # For OLS, positive coef means higher logged damages for more-feminine names -> supports hypothesis
        supports = None
        if stats["pvalue"] is not None:
            supports = (stats["coef"] > 0) and (stats["pvalue"] < sig_alpha)
        else:
            supports = (stats["coef"] > 0)
        stats["supports_hypothesis"] = supports
        out["damages"][var] = stats

    # Build a short human-readable description summarizing results for the focal variables
    def summarize_section(section_name, section_dict, effect_type):
        lines = []
        for var, s in section_dict.items():
            if s is None:
                lines.append(f"{section_name} - {var}: not available in model.")
                continue
            if effect_type == "irr":
                lines.append(
                    f"{section_name} - {var}: coef={s['coef']:.4f}, IRR={s['irr']:.4f} "
                    f"(95% CI IRR [{s['irr_95ci'][0]:.4f}, {s['irr_95ci'][1]:.4f}] ), "
                    f"p={s['pvalue']:.3f} -> "
                    f"{'supports' if s['supports_hypothesis'] else 'does NOT support'} the hypothesis."
                )
            else:
                lines.append(
                    f"{section_name} - {var}: coef={s['coef']:.4f} (95% CI [{s['95ci'][0]:.4f}, {s['95ci'][1]:.4f}]), "
                    f"p={s['pvalue']:.3f} -> "
                    f"{'supports' if s['supports_hypothesis'] else 'does NOT support'} the hypothesis."
                )
        return lines

    description_lines += summarize_section("Deaths (Negative Binomial)", out["deaths"], effect_type="irr")
    description_lines += summarize_section("Damages (OLS on log)", out["damages"], effect_type="coef")

    overall_description = (
        "Interpretation guidance: the hypothesis is that more-feminine hurricane names lead to "
        "fewer precautions, producing MORE fatalities and HIGHER damages. Therefore a positive "
        "coefficient (or IRR > 1 for the death NB model) that is statistically significant "
        "(p < 0.05) counts as evidence supporting the hypothesis. The entries below report "
        "coefficients, 95% CIs, p-values, and whether each focal variable's effect supports the hypothesis."
    )

    description = overall_description + "\n\n" + "\n".join(description_lines)

    return {"object": out, "description": description}