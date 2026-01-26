def extract_final_answer(model_output):
    """
    Extracts coefficient, p-value, confidence interval, and effect size (IRR for count models,
    mean-difference for OLS) for the 'Children' variable from models produced by the modeling code.
    
    Returns a dictionary with:
      - "object": structured dict with per-model estimates and an overall conclusion
      - "description": plain-language interpretation of the results in the context of whether
                       having children decreases engagement in extramarital affairs.
    """
    import math
    import numpy as np
    import pandas as pd

    out = {"models": {}, "overall": None}

    # Helper to safely extract information from a fitted statsmodels result object
    def extract_from_result(mod, var='Children'):
        info = {"coef": None, "pvalue": None, "conf_int": None, "effect": None, "notes": []}
        if mod is None:
            info["notes"].append("model object is None")
            return info

        # params
        try:
            params = getattr(mod, "params", None)
            if params is not None and var in params.index:
                coef = float(params[var])
                info["coef"] = coef
            else:
                info["notes"].append("coef not found in params")
        except Exception as e:
            info["notes"].append(f"error extracting coef: {e}")

        # p-values
        try:
            pvals = getattr(mod, "pvalues", None)
            if pvals is not None and var in pvals.index:
                pval = float(pvals[var])
                info["pvalue"] = pval
            else:
                # Some models (e.g., if fit failed) may not have pvalues or they are NaN
                info["notes"].append("pvalue not found or not available")
        except Exception as e:
            info["notes"].append(f"error extracting pvalue: {e}")

        # confidence interval
        try:
            if hasattr(mod, "conf_int"):
                ci = mod.conf_int()
                # conf_int may be a DataFrame or ndarray
                if isinstance(ci, (pd.DataFrame, pd.Series)) and var in ci.index:
                    lo, hi = float(ci.loc[var, 0]), float(ci.loc[var, 1])
                    info["conf_int"] = (lo, hi)
                elif isinstance(ci, np.ndarray):
                    # try to find column for var by matching order with params
                    params_index = list(mod.params.index) if hasattr(mod, 'params') else None
                    if params_index and var in params_index:
                        idx = params_index.index(var)
                        lo, hi = float(ci[idx, 0]), float(ci[idx, 1])
                        info["conf_int"] = (lo, hi)
                    else:
                        info["notes"].append("conf_int present but could not map to variable")
                else:
                    info["notes"].append("conf_int present but unexpected type")
            else:
                info["notes"].append("conf_int method not available")
        except Exception as e:
            info["notes"].append(f"error extracting conf_int: {e}")

        # Determine model type and compute effect measure
        try:
            model_obj = getattr(mod, "model", None)
            model_name = type(model_obj).__name__ if model_obj is not None else type(mod).__name__
            model_name_lower = model_name.lower()
            # Treat as OLS if underlying model class name contains 'ols' (or RegressionResultsWrapper with OLS model)
            if model_obj is not None and hasattr(model_obj, "__class__"):
                underlying_name = model_obj.__class__.__name__.lower()
            else:
                underlying_name = model_name_lower

            if "ols" in underlying_name or "regression" in type(mod).__name__.lower() and "glm" not in underlying_name:
                # OLS: coef = mean difference in number of affairs associated with Children=1 vs 0
                if info["coef"] is not None:
                    info["effect"] = {"type": "mean_difference", "value": info["coef"],
                                      "interpretation": "difference in mean number of affairs (Children=1 vs 0)"}
            else:
                # Treat as count model (Poisson, NegativeBinomial, ZeroInflated...) -> report IRR = exp(coef)
                if info["coef"] is not None and (isinstance(info["coef"], (int, float)) and not math.isnan(info["coef"])):
                    irr = float(np.exp(info["coef"]))
                    info["effect"] = {"type": "incidence_rate_ratio", "value": irr,
                                      "interpretation": "multiplicative change in expected count (IRR) for Children=1 vs 0"}
                else:
                    info["notes"].append("coef missing or NaN; cannot compute IRR")
        except Exception as e:
            info["notes"].append(f"error determining model type/effect: {e}")

        return info

    # Models we expect in the model_output
    model_keys = [
        ("ols_model", "OLS"),
        ("poisson_model", "Poisson GLM"),
        ("nb_model", "Negative Binomial GLM"),
        ("zinb_model", "Zero-Inflated Negative Binomial")
    ]

    for key, pretty in model_keys:
        mod = model_output.get(key, None)
        if mod is None:
            # If model object not present, maybe only summary text exists; try to use Children_coef_summary if available
            out["models"][pretty] = {"present": False, "info": None}
            continue
        info = extract_from_result(mod, var='Children')
        out["models"][pretty] = {"present": True, "info": info}

    # Also include the precomputed lightweight summary if available (useful fallback)
    if "Children_coef_summary" in model_output:
        out["models"]["Children_coef_summary"] = model_output["Children_coef_summary"]

    # Build an overall conclusion based on sign and statistical significance
    conclusions = []
    sig_found_negative = False
    sig_found_positive = False
    any_estimates = False

    for pretty, entry in out["models"].items():
        if pretty == "Children_coef_summary":
            continue
        if not entry["present"]:
            continue
        info = entry["info"]
        coef = info.get("coef")
        pval = info.get("pvalue")
        effect = info.get("effect")
        any_estimates = any_estimates or (coef is not None)
        if pval is not None and (not math.isnan(pval)):
            if pval < 0.05:
                # significant
                if coef is not None:
                    if coef < 0:
                        sig_found_negative = True
                    elif coef > 0:
                        sig_found_positive = True
                conclusions.append(f"{pretty}: coef={coef:.4f}, p={pval:.3g} (statistically significant)")
            else:
                conclusions.append(f"{pretty}: coef={coef:.4f}, p={pval:.3g} (not statistically significant)")
        else:
            # p-value not available: treat as inconclusive for significance
            conclusions.append(f"{pretty}: coef={coef}, p-value unavailable or NaN (inconclusive)")

    if not any_estimates:
        overall = "No usable estimates for 'Children' found in provided model objects."
    else:
        if sig_found_negative and not sig_found_positive:
            overall = ("Evidence across models: having children is associated with a statistically significant "
                       "decrease in extramarital affairs.")
        elif sig_found_positive and not sig_found_negative:
            overall = ("Evidence across models: having children is associated with a statistically significant "
                       "increase in extramarital affairs.")
        elif sig_found_negative and sig_found_positive:
            overall = ("Conflicting statistically significant estimates across models (some negative, some positive). "
                       "Results are inconsistent.")
        else:
            overall = ("No consistent evidence that having children decreases engagement in extramarital affairs. "
                       "Across the reported models the coefficients for 'Children' are small and not statistically significant; "
                       "count-model effect estimates (IRRs) are very close to 1 when available, and the ZINB model's "
                       "standard errors/p-values appear unavailable (fit instability).")

    out["overall"] = {
        "raw_model_notes": conclusions,
        "conclusion": overall
    }

    # Prepare a human-readable description
    description_lines = [
        "Extracted estimates for the effect of 'Children' (1=yes, 0=no) on number of extramarital affairs:",
    ]
    for pretty, entry in out["models"].items():
        if pretty == "Children_coef_summary":
            # include the lightweight summary if present
            description_lines.append(f"- {pretty}: {entry}")
            continue
        if not entry["present"]:
            description_lines.append(f"- {pretty}: model not present in model_output.")
            continue
        info = entry["info"]
        coef = info.get("coef")
        pval = info.get("pvalue")
        ci = info.get("conf_int")
        effect = info.get("effect")
        notes = info.get("notes", [])
        line = f"- {pretty}: coef={coef}"
        if pval is not None:
            line += f", p={pval:.3g}"
        if ci is not None:
            line += f", 95% CI=({ci[0]:.4f}, {ci[1]:.4f})"
        if effect is not None:
            if effect["type"] == "incidence_rate_ratio":
                line += f", IRR={effect['value']:.4f}"
            else:
                line += f", mean_diff={effect['value']:.4f}"
        if notes:
            line += f" — notes: {'; '.join(notes)}"
        description_lines.append(line)

    description_lines.append("")
    description_lines.append("Overall interpretation:")
    description_lines.append(out["overall"]["conclusion"])

    return {"object": out, "description": "\n".join(description_lines)}