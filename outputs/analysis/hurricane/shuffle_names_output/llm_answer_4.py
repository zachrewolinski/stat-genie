def extract_final_answer(model_output):
    import numpy as np
    import pandas as pd

    res = model_output

    # Handle explicit fallback/no-data case set by the modeling function
    if getattr(res, "_no_observations_fallback", False):
        return {
            "object": None,
            "description": "No observations with non-missing ndam15 were available; an intercept-only fallback model was fit. No coefficient for name femininity is available."
        }

    # Try to extract parameter estimates
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()  # DataFrame with columns [0,1]
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract results from model_output: {e}"
        }

    results = {}
    desc_lines = []

    for var in ["name_c", "is_female_name"]:
        if var in params.index:
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else None
            pval = float(pvalues[var]) if var in pvalues.index else None
            ci_low, ci_high = float(conf.loc[var, 0]), float(conf.loc[var, 1])
            irr = float(np.exp(coef))
            irr_ci_low, irr_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

            results[var] = {
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "incidence_rate_ratio": irr,
                "irr_95_lower": irr_ci_low,
                "irr_95_upper": irr_ci_high
            }

            # Interpretation line for this variable
            if irr < 1:
                direction = ("higher femininity is associated with fewer expected deaths "
                             f"(IRR={irr:.3f} < 1)")
            elif irr > 1:
                direction = ("higher femininity is associated with more expected deaths "
                             f"(IRR={irr:.3f} > 1)")
            else:
                direction = "no multiplicative change in expected deaths (IRR ≈ 1)."

            desc_lines.append(
                f"{var}: coef={coef:.4f}, SE={se:.4f}, p={pval:.4g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
                f"IRR={irr:.4f}, 95% CI [{irr_ci_low:.4f}, {irr_ci_high:.4f}]. Interpretation: {direction}."
            )

    if not results:
        return {
            "object": None,
            "description": "Neither 'name_c' nor 'is_female_name' are present in the fitted model; nothing to extract."
        }

    # Produce a brief overall interpretation focused on the primary variable (name_c if present)
    primary = "name_c" if "name_c" in results else "is_female_name"
    primary_res = results[primary]
    p = primary_res["p_value"]
    significance = "statistically significant" if p is not None and p < 0.05 else "not statistically significant (at alpha=0.05)"
    overall_interpretation = (
        f"Primary result ({primary}): a one-unit increase in {primary} multiplies expected hurricane deaths by "
        f"{primary_res['incidence_rate_ratio']:.3f} (95% CI {primary_res['irr_95_lower']:.3f}–{primary_res['irr_95_upper']:.3f}); "
        f"p = {p:.4g}, {significance}. "
        "In context: IRR < 1 would support the hypothesis that more feminine names are perceived as less threatening (fewer deaths); "
        "IRR > 1 would indicate the opposite. This is an association from an observational regression controlling for listed covariates, "
        "not by itself proof of causation."
    )

    description = "\n".join(desc_lines) + "\n\n" + overall_interpretation

    return {"object": results, "description": description}