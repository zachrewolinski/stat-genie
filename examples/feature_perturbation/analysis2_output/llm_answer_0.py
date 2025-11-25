def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and incidence-rate ratios (IRRs)
    for the name-femininity predictors from a fitted statsmodels GLM (NegativeBinomial) results object.

    Returns a dictionary:
      {
        "object": { variable_name: {coef, se, pvalue, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, perc_change, perc_change_ci}, ... },
        "description": "Textual interpretation of the results in context"
      }
    """
    import numpy as np

    # Variables of interest
    vars_of_interest = ["FeminineName", "NameFemininity_z"]

    res = {}
    try:
        params = model_output.params          # pandas Series
        bse = model_output.bse
        pvals = model_output.pvalues
        ci = model_output.conf_int()          # DataFrame: columns [0,1] or named
    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting results from model_output: {e}"
        }

    for v in vars_of_interest:
        if v not in params.index:
            # Variable not in model (e.g., dropped or not included)
            res[v] = None
            continue

        coef = float(params.loc[v])
        se = float(bse.loc[v]) if v in bse.index else None
        pval = float(pvals.loc[v]) if v in pvals.index else None

        # Confidence interval
        try:
            ci_row = ci.loc[v]
            # conf_int returns a 2-column array/dataframe; ensure ordering
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
        except Exception:
            ci_lower = None
            ci_upper = None

        # For count models (log link), exponentiate coefficient to get IRR (incidence rate ratio)
        try:
            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            irr = None
            irr_ci_lower = None
            irr_ci_upper = None

        # Percentage change in expected deaths associated with one-unit increase (or binary change)
        perc_change = (irr - 1) * 100 if irr is not None else None
        perc_change_ci_lower = (irr_ci_lower - 1) * 100 if irr_ci_lower is not None else None
        perc_change_ci_upper = (irr_ci_upper - 1) * 100 if irr_ci_upper is not None else None

        # Significance at conventional levels
        sig_05 = (pval is not None) and (pval < 0.05)

        res[v] = {
            "coef": round(coef, 4),
            "se": round(se, 4) if se is not None else None,
            "pvalue": round(pval, 4) if pval is not None else None,
            "ci_lower": round(ci_lower, 4) if ci_lower is not None else None,
            "ci_upper": round(ci_upper, 4) if ci_upper is not None else None,
            "irr": round(irr, 4) if irr is not None else None,
            "irr_ci_lower": round(irr_ci_lower, 4) if irr_ci_lower is not None else None,
            "irr_ci_upper": round(irr_ci_upper, 4) if irr_ci_upper is not None else None,
            "perc_change": round(perc_change, 2) if perc_change is not None else None,
            "perc_change_ci": (round(perc_change_ci_lower, 2), round(perc_change_ci_upper, 2)) if (perc_change_ci_lower is not None and perc_change_ci_upper is not None) else None,
            "significant_p_lt_0.05": bool(sig_05)
        }

    # Create a short interpretation for the primary test (binary FeminineName)
    if res.get("FeminineName") is None:
        interpretation = "The model does not contain the variable 'FeminineName' (it may have been dropped)."
    else:
        vres = res["FeminineName"]
        if vres is None:
            interpretation = "The variable 'FeminineName' is not available in the fitted model results."
        else:
            if vres["irr"] is None:
                interpretation = "Could not compute IRR for 'FeminineName'."
            else:
                # Direction and significance
                direction = "lower" if vres["irr"] < 1 else "higher"
                sig_text = "statistically significant (p < 0.05)" if vres["significant_p_lt_0.05"] else "not statistically significant (p >= 0.05)"
                interpretation = (
                    f"Binary FeminineName -> coef = {vres['coef']}, p = {vres['pvalue']}. "
                    f"IRR = {vres['irr']} (95% CI: {vres['irr_ci_lower']} to {vres['irr_ci_upper']}). "
                    f"This implies that hurricanes given feminine names are associated with {abs(vres['perc_change']):.1f}% {direction} expected deaths "
                    f"relative to masculine-named hurricanes, and this effect is {sig_text}."
                )

    return {
        "object": res,
        "description": interpretation
    }