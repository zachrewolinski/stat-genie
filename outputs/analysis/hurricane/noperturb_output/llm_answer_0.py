def extract_final_answer(model_output):
    """
    Extracts the coefficient and inference for the 'masfem_z' variable from a fitted
    statsmodels results object (GLM/GLMResultsWrapper or robust wrapper).
    
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI, IRR, IRR 95% CI, nobs)
      - "description": human-readable interpretation in the context of the hypothesis:
                       "more feminine names -> more fatalities" (supports / does not support / inconclusive)
    """
    import numpy as np
    from math import erf, sqrt

    var = 'masfem_z'
    out = {
        "coef": None,
        "std_err": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "irr": None,
        "irr_ci_lower": None,
        "irr_ci_upper": None,
        "nobs": None
    }

    # helper to compute p-value from z-statistic (two-sided), avoids external dependencies
    def p_from_z(z):
        return 2 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

    # try to extract values robustly from common statsmodels result objects
    try:
        coef = float(model_output.params[var])
        out["coef"] = coef
    except Exception:
        raise KeyError(f"Could not find parameter '{var}' in model_output.params")

    # standard error
    try:
        se = float(model_output.bse[var])
        out["std_err"] = se
    except Exception:
        # if bse missing, leave as None
        se = None

    # p-value (if present)
    try:
        pval = float(model_output.pvalues[var])
        out["p_value"] = pval
    except Exception:
        # compute from z if possible
        if se is not None and se > 0:
            z = coef / se
            pval = p_from_z(z)
            out["p_value"] = pval
        else:
            out["p_value"] = None

    # confidence interval
    try:
        ci = model_output.conf_int()
        # conf_int may be DataFrame or ndarray. Try to index by variable name.
        if hasattr(ci, "loc"):
            ci_row = ci.loc[var]
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        else:
            # assume array-like with same ordering as params index
            params_index = list(model_output.params.index)
            idx = params_index.index(var)
            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
        out["ci_lower"], out["ci_upper"] = ci_lower, ci_upper
    except Exception:
        # fallback: compute Wald CI using normal approx (z_crit ~ 1.96)
        if se is not None:
            zcrit = 1.959963984540054
            ci_lower = coef - zcrit * se
            ci_upper = coef + zcrit * se
            out["ci_lower"], out["ci_upper"] = ci_lower, ci_upper
        else:
            out["ci_lower"], out["ci_upper"] = None, None

    # IRR (exp(coef)) and its CI
    try:
        if out["coef"] is not None:
            irr = float(np.exp(out["coef"]))
            out["irr"] = irr
            if out["ci_lower"] is not None and out["ci_upper"] is not None:
                out["irr_ci_lower"] = float(np.exp(out["ci_lower"]))
                out["irr_ci_upper"] = float(np.exp(out["ci_upper"]))
    except Exception:
        out["irr"], out["irr_ci_lower"], out["irr_ci_upper"] = None, None, None

    # nobs
    try:
        # try common attributes
        if hasattr(model_output, 'nobs'):
            out["nobs"] = int(model_output.nobs)
        elif hasattr(model_output, 'model') and hasattr(model_output.model, 'nobs'):
            out["nobs"] = int(model_output.model.nobs)
        elif hasattr(model_output, 'model') and hasattr(model_output.model, 'endog'):
            out["nobs"] = int(getattr(model_output.model.endog, "shape", (len(model_output.model.endog),))[0])
    except Exception:
        out["nobs"] = None

    # Simple interpretation relative to the hypothesis:
    # Hypothesis: more feminine names -> fewer precautions -> higher fatalities
    # i.e., positive coef (and statistically significant) supports the hypothesis.
    coef_val = out["coef"]
    pval_val = out["p_value"]
    desc_lines = []
    desc_lines.append(f"Variable: {var}")
    desc_lines.append(f"Estimated coefficient (log-IRR): {coef_val}")
    if out["std_err"] is not None:
        desc_lines.append(f"Standard error: {out['std_err']}")
    if pval_val is not None:
        desc_lines.append(f"Two-sided p-value: {pval_val}")
    if out["ci_lower"] is not None and out["ci_upper"] is not None:
        desc_lines.append(f"95% CI (log scale): [{out['ci_lower']}, {out['ci_upper']}]")
    if out["irr"] is not None:
        desc_lines.append(f"Incidence Rate Ratio (IRR) = exp(coef): {out['irr']}")
    if out["irr_ci_lower"] is not None and out["irr_ci_upper"] is not None:
        desc_lines.append(f"95% CI for IRR: [{out['irr_ci_lower']}, {out['irr_ci_upper']}]")
    if out["nobs"] is not None:
        desc_lines.append(f"Number of observations (approx): {out['nobs']}")

    # Decision text
    decision = "Inconclusive"
    if coef_val is not None:
        if pval_val is not None:
            alpha = 0.05
            if coef_val > 0 and pval_val < alpha:
                decision = "Supports hypothesis (positive coef, statistically significant)"
            elif coef_val > 0 and pval_val >= alpha:
                decision = "Does not provide statistically significant evidence for the hypothesis (positive coef but not significant)"
            elif coef_val <= 0 and pval_val < alpha:
                decision = "Contradicts hypothesis (negative or zero coef, statistically significant)"
            else:
                decision = "Does not provide statistically significant evidence; coefficient not significantly different from zero"
        else:
            # no p-value available; base on sign only
            if coef_val > 0:
                decision = "Positive coefficient (no p-value available) — suggestive but not conclusive"
            elif coef_val <= 0:
                decision = "Non-positive coefficient (no p-value available) — does not support hypothesis"
    desc_lines.append("Conclusion: " + decision)

    description = "; ".join(desc_lines)

    return {"object": out, "description": description}