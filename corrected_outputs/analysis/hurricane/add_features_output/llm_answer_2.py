def extract_final_answer(model_output):
    """
    Extract statistics about the effect of name femininity on fatalities from the models
    returned by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing extracted numbers (coefficients, SEs, p-values,
                  confidence intervals, effect sizes) for 'masfem_z' and 'female_name'
                  from the primary OLS on log_deaths and the Negative Binomial on counts,
                  and an overall decision based on the primary OLS result.
      - "description": a brief interpretation of those numbers in the context of the
                       hypothesis that more feminine hurricane names lead to fewer precautions
                       (operationalized as higher fatalities).

    Decision rule (automated, based on the primary OLS on log_deaths):
      - "supports": if masfem_z coefficient > 0 and p < 0.05 (feminine names associated with more deaths)
      - "contradicts": if masfem_z coefficient < 0 and p < 0.05 (feminine names associated with fewer deaths)
      - "inconclusive": otherwise (insufficient evidence)
    """
    import math
    import numpy as np

    out = {"masfem_z": {"ols": None, "nb": None},
           "female_name": {"ols": None, "nb": None},
           "decision_basis": None}

    def safe_extract_ols(res, var):
        if res is None:
            return None
        try:
            params = res.params
            if var not in params.index:
                return None
            coef = float(params[var])
            se = float(res.bse[var]) if var in res.bse.index else None
            p = float(res.pvalues[var]) if var in res.pvalues.index else None
            try:
                ci = res.conf_int().loc[var].tolist()
            except Exception:
                ci = None
            # For log outcome: percent change approx = (exp(coef) - 1) * 100
            pct = (math.exp(coef) - 1) * 100 if coef is not None else None
            pct_ci = None
            if ci is not None:
                pct_ci = [(math.exp(ci[0]) - 1) * 100, (math.exp(ci[1]) - 1) * 100]
            return {"coef": coef, "se": se, "pvalue": p, "ci": ci, "pct_change": pct, "pct_ci": pct_ci}
        except Exception:
            return None

    def safe_extract_glm(res, var):
        if res is None:
            return None
        try:
            params = res.params
            if var not in params.index:
                return None
            coef = float(params[var])
            se = float(res.bse[var]) if var in res.bse.index else None
            p = float(res.pvalues[var]) if hasattr(res, "pvalues") and var in res.pvalues.index else None
            try:
                ci = res.conf_int().loc[var].tolist()
            except Exception:
                ci = None
            # For log link GLM (NegativeBinomial/Poisson): exp(coef) = incidence rate ratio
            irr = math.exp(coef) if coef is not None else None
            irr_ci = [math.exp(ci[0]), math.exp(ci[1])] if ci is not None else None
            return {"coef": coef, "se": se, "pvalue": p, "ci": ci, "irr": irr, "irr_ci": irr_ci}
        except Exception:
            return None

    # Primary (preferred) model: OLS on log_deaths
    ols = model_output.get("ols_log_deaths")
    nb = model_output.get("nb_deaths")

    # Extract for masfem_z and female_name from OLS and NB (if present)
    for var in ("masfem_z", "female_name"):
        out["masfem_z" if var == "masfem_z" else "female_name"]["ols"] = safe_extract_ols(ols, var)
        out["masfem_z" if var == "masfem_z" else "female_name"]["nb"] = safe_extract_glm(nb, var)

    # Decision based on primary OLS masfem_z estimate (if available)
    decision = {"decision": "inconclusive", "reason": None}
    ols_m = out["masfem_z"]["ols"]
    if ols_m is None:
        # try female_name as fallback
        ols_f = out["female_name"]["ols"]
        if ols_f is None:
            decision["decision"] = "inconclusive"
            decision["reason"] = "Neither 'masfem_z' nor 'female_name' present in the OLS model results."
        else:
            coef = ols_f.get("coef")
            p = ols_f.get("pvalue")
            if coef is None or p is None:
                decision["decision"] = "inconclusive"
                decision["reason"] = "Could not extract coefficient/p-value for 'female_name' from OLS."
            else:
                if (p < 0.05) and (coef > 0):
                    decision["decision"] = "supports"
                    decision["reason"] = ("'female_name' (binary) has a positive, statistically significant "
                                          "coefficient in OLS, indicating historically female names are "
                                          "associated with higher fatalities (consistent with hypothesis).")
                elif (p < 0.05) and (coef < 0):
                    decision["decision"] = "contradicts"
                    decision["reason"] = ("'female_name' (binary) has a negative, statistically significant "
                                          "coefficient in OLS (opposite of hypothesis).")
                else:
                    decision["decision"] = "inconclusive"
                    decision["reason"] = ("'female_name' coefficient is not statistically significant in OLS "
                                          "(p >= 0.05); evidence is inconclusive.")
    else:
        coef = ols_m.get("coef")
        p = ols_m.get("pvalue")
        if coef is None or p is None:
            decision["decision"] = "inconclusive"
            decision["reason"] = "Could not extract coefficient/p-value for 'masfem_z' from OLS."
        else:
            if (p < 0.05) and (coef > 0):
                decision["decision"] = "supports"
                decision["reason"] = ("A 1 SD increase in name femininity (masfem_z) is associated with a "
                                      f"{ols_m.get('pct_change'):.1f}% increase in expected fatalities "
                                      f"(OLS coef = {coef:.4f}, p = {p:.3f}). This is consistent with the hypothesis "
                                      "that more feminine names lead to fewer precautions and thus more deaths.")
            elif (p < 0.05) and (coef < 0):
                decision["decision"] = "contradicts"
                decision["reason"] = ("A 1 SD increase in name femininity (masfem_z) is associated with a "
                                      f"{ols_m.get('pct_change'):.1f}% decrease in expected fatalities "
                                      f"(OLS coef = {coef:.4f}, p = {p:.3f}) — opposite the hypothesis.")
            else:
                decision["decision"] = "inconclusive"
                decision["reason"] = ("masfem_z coefficient in OLS is not statistically significant (p >= 0.05); "
                                      "evidence is inconclusive about the hypothesized effect.")

    out["decision_basis"] = decision

    # Human-readable description summarizing what the numbers mean
    description_parts = []
    description_parts.append("Extracted statistics relate to whether more feminine hurricane names are associated with higher fatalities (the hypothesis).")
    if out["masfem_z"]["ols"] is not None:
        m = out["masfem_z"]["ols"]
        description_parts.append(
            f"OLS on log_deaths — masfem_z: coef = {m['coef']:.4f}, se = {m['se']:.4f}, p = {m['pvalue']:.3f}, "
            f"95% CI = {m['ci']}, approximate % change in deaths per 1 SD = {m['pct_change']:.1f}% "
            f"(95% CI = {m['pct_ci']})."
        )
    else:
        description_parts.append("masfem_z not available in OLS results.")

    if out["masfem_z"]["nb"] is not None:
        m = out["masfem_z"]["nb"]
        description_parts.append(
            f"Negative Binomial on counts — masfem_z: coef = {m['coef']:.4f}, se = {m['se']:.4f}, p = {m['pvalue']}, "
            f"IRR = {m['irr']:.3f}, IRR 95% CI = {m['irr_ci']}."
        )
    else:
        description_parts.append("masfem_z not available in Negative Binomial results (or extraction failed).")

    # Add final decision summary
    dec = out["decision_basis"]
    description_parts.append(f"Automated decision (based on primary OLS masfem_z): {dec['decision']}. {dec['reason']}")

    return {"object": out, "description": " ".join(description_parts)}