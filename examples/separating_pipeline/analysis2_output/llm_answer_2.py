def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and transformed effect
    measures (IRR for count models) for the focal predictors in the provided model_output dict.
    
    Expects model_output to be a dict with keys:
      - 'nb_masfem' : statsmodels GLMResultsWrapper (NegativeBinomial/Poisson)
      - 'nb_femalebinary' : statsmodels GLMResultsWrapper (NegativeBinomial/Poisson)
      - 'ols_logdeath' : statsmodels OLSResults (with robust SEs if fit that way)
    
    Returns a dict: { "object": <summary dict>, "description": <text interpretation> }
    """
    import numpy as np
    import pandas as pd

    def _safe_extract(model, var):
        """Extract param, se, pval, CI for var from a statsmodels results object."""
        out = {"var": var, "present": False}
        try:
            params = model.params
            if var not in params.index:
                return out
            out["present"] = True
            coef = float(params[var])
            out["coef"] = coef
        except Exception:
            return out

        # standard error and p-value (if available)
        try:
            out["se"] = float(model.bse[var])
        except Exception:
            out["se"] = None
        try:
            out["pvalue"] = float(model.pvalues[var])
        except Exception:
            out["pvalue"] = None

        # 95% CI
        try:
            ci = model.conf_int()
            # conf_int() may return DataFrame or ndarray
            if isinstance(ci, pd.DataFrame):
                ci_lower = float(ci.loc[var].iloc[0])
                ci_upper = float(ci.loc[var].iloc[1])
            else:
                # assume ndarray with same order as params.index
                idx = list(model.params.index).index(var)
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
            out["ci_lower"] = ci_lower
            out["ci_upper"] = ci_upper
        except Exception:
            out["ci_lower"] = out["ci_upper"] = None

        return out

    summary = {}

    # 1) Negative binomial model with MasFem_z
    nb_mas = model_output.get("nb_masfem")
    summary["nb_masfem"] = _safe_extract(nb_mas, "MasFem_z") if nb_mas is not None else {"error": "nb_masfem missing"}

    # If this is a count model (GLM with log link), exponentiate coef to get IRR
    try:
        if summary["nb_masfem"].get("present", False):
            coef = summary["nb_masfem"]["coef"]
            summary["nb_masfem"]["IRR"] = float(np.exp(coef))
            if summary["nb_masfem"]["ci_lower"] is not None:
                summary["nb_masfem"]["IRR_ci_lower"] = float(np.exp(summary["nb_masfem"]["ci_lower"]))
                summary["nb_masfem"]["IRR_ci_upper"] = float(np.exp(summary["nb_masfem"]["ci_upper"]))
    except Exception:
        pass

    # 2) Negative binomial model with FemaleName
    nb_fem = model_output.get("nb_femalebinary")
    summary["nb_femalebinary"] = _safe_extract(nb_fem, "FemaleName") if nb_fem is not None else {"error": "nb_femalebinary missing"}
    try:
        if summary["nb_femalebinary"].get("present", False):
            coef = summary["nb_femalebinary"]["coef"]
            summary["nb_femalebinary"]["IRR"] = float(np.exp(coef))
            if summary["nb_femalebinary"]["ci_lower"] is not None:
                summary["nb_femalebinary"]["IRR_ci_lower"] = float(np.exp(summary["nb_femalebinary"]["ci_lower"]))
                summary["nb_femalebinary"]["IRR_ci_upper"] = float(np.exp(summary["nb_femalebinary"]["ci_upper"]))
    except Exception:
        pass

    # 3) OLS on log-deaths (has both MasFem_z and FemaleName in formula)
    ols = model_output.get("ols_logdeath")
    summary["ols_logdeath"] = {}
    if ols is None:
        summary["ols_logdeath"]["error"] = "ols_logdeath missing"
    else:
        for var in ["MasFem_z", "FemaleName"]:
            summary["ols_logdeath"][var] = _safe_extract(ols, var)

        # For log outcome, coefficient approx. equals proportional change in (Deaths+1):
        # multiply by 100 to get approximate percent change per unit.
        for var in ["MasFem_z", "FemaleName"]:
            entry = summary["ols_logdeath"].get(var, {})
            if entry.get("present", False) and entry.get("coef") is not None:
                entry["approx_pct_change"] = entry["coef"] * 100.0
                if entry.get("ci_lower") is not None:
                    entry["approx_pct_change_ci_lower"] = entry["ci_lower"] * 100.0
                    entry["approx_pct_change_ci_upper"] = entry["ci_upper"] * 100.0

    # Make a simple conclusion about the hypothesis:
    # Hypothesis: "More feminine names -> perceived less threatening -> fewer fatalities"
    # Operational test: negative coefficient on MasFem_z (and FemaleName=1) and statistically significant (p < 0.05).
    def _conclude(entry, model_type, varname):
        if not entry.get("present", False):
            return {"conclusion": "variable_missing", "reason": f"{varname} missing in {model_type}"}
        coef = entry.get("coef")
        p = entry.get("pvalue")
        if p is None:
            return {"conclusion": "no_pvalue", "reason": "p-value unavailable"}
        if coef < 0 and p < 0.05:
            return {"conclusion": "supports_hypothesis", "reason": f"{varname} coef < 0 and p={p:.3g} < 0.05"}
        if coef > 0 and p < 0.05:
            return {"conclusion": "contradicts_hypothesis", "reason": f"{varname} coef > 0 and p={p:.3g} < 0.05"}
        return {"conclusion": "inconclusive", "reason": f"{varname} coef={coef:.4g}, p={p:.3g} (not < 0.05)"}

    conclusions = {
        "nb_masfem": _conclude(summary["nb_masfem"], "nb_masfem", "MasFem_z") if isinstance(summary.get("nb_masfem"), dict) else {"error": "missing"},
        "nb_femalebinary": _conclude(summary["nb_femalebinary"], "nb_femalebinary", "FemaleName") if isinstance(summary.get("nb_femalebinary"), dict) else {"error": "missing"},
        "ols_masfem": _conclude(summary["ols_logdeath"].get("MasFem_z", {}), "ols_logdeath", "MasFem_z"),
        "ols_femalebinary": _conclude(summary["ols_logdeath"].get("FemaleName", {}), "ols_logdeath", "FemaleName")
    }

    # Build a short textual description summarizing the numeric output and the conclusion logic.
    description_lines = [
        "Extracted statistics for focal predictors from each model. For GLM count models (nb_*), coef is on the log scale;",
        "IRR = exp(coef) is the multiplicative change in expected deaths per unit increase in the predictor.",
        "For OLS on log(Deaths+1), coef approximates proportional change; coef*100 ~ approximate percent change in (Deaths+1).",
        "",
        "Summary of extracted values and statistical conclusions (decision rule: p < 0.05 and negative coef -> supports hypothesis):",
        f" - nb_masfem (MasFem_z): {conclusions['nb_masfem']['conclusion']} ({conclusions['nb_masfem']['reason']})",
        f" - nb_femalebinary (FemaleName): {conclusions['nb_femalebinary']['conclusion']} ({conclusions['nb_femalebinary']['reason']})",
        f" - ols_logdeath MasFem_z: {conclusions['ols_masfem']['conclusion']} ({conclusions['ols_masfem']['reason']})",
        f" - ols_logdeath FemaleName: {conclusions['ols_femalebinary']['conclusion']} ({conclusions['ols_femalebinary']['reason']})",
        "",
        "Use the numeric 'object' contents to report exact coefficients, p-values, confidence intervals, IRRs, and percent-change approximations."
    ]

    return {
        "object": {
            "summary": summary,
            "conclusions": conclusions
        },
        "description": "\n".join(description_lines)
    }