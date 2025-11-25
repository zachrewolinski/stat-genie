def extract_final_answer(model_output):
    """
    Extract key statistics for the femininity predictor from fitted model objects.
    Expects model_output to be the dict returned by the modeling function, with keys
    like 'ols_masfem', 'ols_gender_binary', 'ols_damage_masfem' (each a statsmodels result).
    Returns a dict with keys:
      - "object": dict of extracted numeric statistics by model
      - "description": human-readable interpretation of the key result(s)
    """
    import math
    import numpy as np

    def _extract_from_result(res_obj, varname):
        if res_obj is None:
            return None
        try:
            params = getattr(res_obj, "params", None)
            pvalues = getattr(res_obj, "pvalues", None)
            bse = getattr(res_obj, "bse", None)
            ci = None
            try:
                ci = res_obj.conf_int()
            except Exception:
                ci = None
            nobs = None
            try:
                nobs = int(getattr(res_obj, "nobs"))
            except Exception:
                nobs = None
            rsq = getattr(res_obj, "rsquared", None)

            if params is None or varname not in params.index:
                # variable not in model
                return None

            coef = float(params[varname])
            se = float(bse[varname]) if (bse is not None and varname in bse.index) else None
            pval = float(pvalues[varname]) if (pvalues is not None and varname in pvalues.index) else None
            ci_low = None
            ci_high = None
            if ci is not None:
                try:
                    # conf_int() returns array-like with index same as params
                    # Try to index by varname if it's a DataFrame/Series
                    if hasattr(ci, "loc"):
                        row = ci.loc[varname]
                        ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
                    else:
                        # fallback: convert to DataFrame-like array and match position
                        # find position of varname in params.index
                        idx = list(params.index).index(varname)
                        ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
                except Exception:
                    ci_low, ci_high = None, None

            # For interpretation on log outcome, compute approximate percent change:
            # percent_change = (exp(coef) - 1) * 100
            try:
                pct_change = (math.exp(coef) - 1.0) * 100.0
            except Exception:
                pct_change = None

            return {
                "coef": coef,
                "se": se,
                "p_value": pval,
                "conf_low": ci_low,
                "conf_high": ci_high,
                "nobs": nobs,
                "rsquared": float(rsq) if rsq is not None else None,
                "approx_pct_change_in_(alldeaths+1)": pct_change,
            }
        except Exception as e:
            return {"error": f"failed to extract from result object: {e}"}

    extracted = {}
    # if model_output is a statsmodels result directly, wrap it
    if not isinstance(model_output, dict):
        # try to treat as single result, assume masfem_std was used
        try:
            extracted['ols_masfem'] = _extract_from_result(model_output, "masfem_std")
        except Exception:
            extracted = {}
    else:
        # expected dict of results
        if "ols_masfem" in model_output:
            extracted["ols_masfem"] = _extract_from_result(model_output.get("ols_masfem"), "masfem_std")
        if "ols_gender_binary" in model_output:
            # gender_mf might be the binary predictor name
            extracted["ols_gender_binary"] = _extract_from_result(model_output.get("ols_gender_binary"), "gender_mf")
        if "ols_damage_masfem" in model_output:
            extracted["ols_damage_masfem"] = _extract_from_result(model_output.get("ols_damage_masfem"), "masfem_std")

    # Prepare a short interpretation
    descriptions = []
    # Interpret ols_masfem first if present
    main = extracted.get("ols_masfem")
    if main is None:
        descriptions.append("No fitted 'ols_masfem' model found or femininity variable not in that model.")
    else:
        if "error" in main:
            descriptions.append(f"Extraction error for ols_masfem: {main['error']}")
        else:
            coef = main.get("coef")
            p = main.get("p_value")
            pct = main.get("approx_pct_change_in_(alldeaths+1)")
            if coef is None:
                descriptions.append("Could not extract coefficient for masfem_std from ols_masfem.")
            else:
                sign = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
                sig = ("statistically significant (p < 0.05)" if (p is not None and p < 0.05) else
                       "not statistically significant (p >= 0.05)" if p is not None else "p-value unavailable")
                desc = (f"Model 'ols_masfem': masfem_std coefficient = {coef:.4g}, SE = {main.get('se'):.4g} if available, "
                        f"p = {p:.4g}." if p is not None else
                        f"Model 'ols_masfem': masfem_std coefficient = {coef:.4g}.")
                desc += f" This is a {sign} association and is {sig}."
                if pct is not None:
                    desc += f" Roughly, a 1 SD increase in name femininity is associated with a {pct:.2f}% change in (alldeaths+1)."
                descriptions.append(desc)

    # Also add brief notes for other models if present
    if "ols_gender_binary" in extracted:
        g = extracted["ols_gender_binary"]
        if g is None:
            descriptions.append("No gender-binary model stats found.")
        elif "error" in g:
            descriptions.append(f"Extraction error for ols_gender_binary: {g['error']}")
        else:
            coef = g.get("coef")
            p = g.get("p_value")
            if coef is not None:
                sign = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
                sig = ("statistically significant (p < 0.05)" if (p is not None and p < 0.05) else
                       "not statistically significant" if p is not None else "p-value unavailable")
                descriptions.append(f"Model 'ols_gender_binary': gender_mf coef = {coef:.4g}, {sign}, {sig}.")
    if "ols_damage_masfem" in extracted:
        d = extracted["ols_damage_masfem"]
        if d is None:
            descriptions.append("No damage robustness model stats found.")
        elif "error" in d:
            descriptions.append(f"Extraction error for ols_damage_masfem: {d['error']}")
        else:
            coef = d.get("coef")
            p = d.get("p_value")
            if coef is not None:
                sign = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
                sig = ("statistically significant (p < 0.05)" if (p is not None and p < 0.05) else
                       "not statistically significant" if p is not None else "p-value unavailable")
                descriptions.append(f"Robustness 'ols_damage_masfem': masfem_std coef = {coef:.4g}, {sign}, {sig}.")

    if not descriptions:
        descriptions = ["No model statistics could be extracted from the provided model_output."]

    return {
        "object": extracted,
        "description": " ".join(descriptions)
    }