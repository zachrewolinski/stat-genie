def extract_final_answer(model_output):
    """
    Extracts coefficient estimates, p-values, and 95% confidence intervals for the
    'beauty' and 'beauty_sq' predictors from the provided model_output dict.
    model_output is expected to have keys 'ols_clustered' and 'mixedlm' whose
    values are fitted statsmodels result objects.

    Returns:
      {
        "object": {
          "ols_clustered": { "beauty": {...}, "beauty_sq": {...} },
          "mixedlm": { "beauty": {...}, "beauty_sq": {...} }
        },
        "description": "Plain-language summary of the estimates and their meaning"
      }
    """
    import numpy as np
    import pandas as pd

    def _safe_get_params(res):
        # Try common attribute names for coefficients, pvalues, bse, conf_int
        params = None
        pvalues = None
        bse = None
        conf_int = None
        tvalues = None

        # params
        if hasattr(res, "params"):
            params = res.params
        elif hasattr(res, "fe_params"):
            params = res.fe_params

        # p-values
        if hasattr(res, "pvalues"):
            pvalues = res.pvalues
        elif hasattr(res, "pvalues_fe"):
            pvalues = res.pvalues_fe

        # standard errors
        if hasattr(res, "bse"):
            bse = res.bse
        elif hasattr(res, "bse_fe"):
            bse = res.bse_fe

        # t/z values
        if hasattr(res, "tvalues"):
            tvalues = res.tvalues
        elif hasattr(res, "tvalues_fe"):
            tvalues = res.tvalues_fe

        # conf_int
        try:
            ci = res.conf_int()
            # conf_int may return ndarray or DataFrame
            if isinstance(ci, (pd.DataFrame, pd.Series)):
                conf_int = ci
            else:
                # convert ndarray to DataFrame with parameter names if possible
                if params is not None:
                    try:
                        conf_int = pd.DataFrame(ci, index=params.index, columns=["ci_lower", "ci_upper"])
                    except Exception:
                        conf_int = pd.DataFrame(ci)
                else:
                    conf_int = pd.DataFrame(ci)
        except Exception:
            conf_int = None

        return params, pvalues, bse, tvalues, conf_int

    def _extract_for_param(res, param_name):
        params, pvalues, bse, tvalues, conf_int = _safe_get_params(res)

        # initialize result dict
        out = {"coef": None, "pvalue": None, "ci_lower": None, "ci_upper": None, "se": None, "stat": None, "significant": None}

        # coefficient
        try:
            if params is not None:
                out["coef"] = float(params[param_name])
        except Exception:
            try:
                # params might be ndarray-like, try to find index
                if params is not None and param_name in getattr(params, "index", []):
                    out["coef"] = float(params.loc[param_name])
            except Exception:
                out["coef"] = None

        # p-value
        try:
            if pvalues is not None and param_name in pvalues.index:
                out["pvalue"] = float(pvalues[param_name])
        except Exception:
            out["pvalue"] = None

        # standard error
        try:
            if bse is not None and param_name in bse.index:
                out["se"] = float(bse[param_name])
        except Exception:
            out["se"] = None

        # test statistic (t or z)
        try:
            if tvalues is not None and param_name in tvalues.index:
                out["stat"] = float(tvalues[param_name])
        except Exception:
            # try compute from coef/se
            if out["coef"] is not None and out["se"] is not None and out["se"] != 0:
                out["stat"] = float(out["coef"] / out["se"])
            else:
                out["stat"] = None

        # confidence interval
        try:
            if conf_int is not None:
                # conf_int could be DataFrame with rows indexed by param name, or 2-col array
                if isinstance(conf_int, pd.DataFrame) and param_name in conf_int.index:
                    # DataFrame might have two columns
                    row = conf_int.loc[param_name]
                    # handle named columns or positional
                    if len(row) >= 2:
                        out["ci_lower"] = float(row.iloc[0])
                        out["ci_upper"] = float(row.iloc[1])
                else:
                    # try to find index of parameter in params and use same row in conf_int
                    if params is not None and hasattr(params, "index"):
                        try:
                            idx = list(params.index).index(param_name)
                            ci_row = conf_int.iloc[idx]
                            out["ci_lower"] = float(ci_row.iloc[0])
                            out["ci_upper"] = float(ci_row.iloc[1])
                        except Exception:
                            pass
        except Exception:
            pass

        # significance (use pvalue if available, otherwise None)
        if out["pvalue"] is not None:
            out["significant"] = bool(out["pvalue"] < 0.05)
        else:
            out["significant"] = None

        return out

    results_object = {"ols_clustered": {}, "mixedlm": {}}

    # Extract from OLS clustered results
    ols_res = model_output.get("ols_clustered", None)
    if ols_res is not None:
        for pname in ["beauty", "beauty_sq"]:
            try:
                results_object["ols_clustered"][pname] = _extract_for_param(ols_res, pname)
            except Exception as e:
                results_object["ols_clustered"][pname] = {"error": str(e)}

    else:
        results_object["ols_clustered"] = None

    # Extract from mixed effects results
    mixed_res = model_output.get("mixedlm", None)
    if mixed_res is not None:
        # MixedLM results often have fixed-effects in .fe_params; create a minimal wrapper
        # Try to extract using the same helper which tries fe_params as fallback
        for pname in ["beauty", "beauty_sq"]:
            try:
                results_object["mixedlm"][pname] = _extract_for_param(mixed_res, pname)
            except Exception as e:
                results_object["mixedlm"][pname] = {"error": str(e)}
    else:
        results_object["mixedlm"] = None

    # Compose a concise description/interpretation
    def _interpret(entry):
        if entry is None:
            return "model not available"
        if "error" in entry:
            return f"error extracting parameter: {entry['error']}"
        coef = entry.get("coef")
        p = entry.get("pvalue")
        ci_l = entry.get("ci_lower")
        ci_u = entry.get("ci_upper")
        sig = entry.get("significant")
        parts = []
        if coef is None:
            parts.append("coefficient unavailable")
        else:
            parts.append(f"coef={coef:.4f}")
        if p is not None:
            parts.append(f"p={p:.3f}")
        else:
            parts.append("p=NA")
        if ci_l is not None and ci_u is not None:
            parts.append(f"95% CI=[{ci_l:.4f}, {ci_u:.4f}]")
        # significance message
        if sig is True:
            parts.append("statistically significant (p<0.05)")
        elif sig is False:
            parts.append("not statistically significant (p>=0.05)")
        else:
            parts.append("significance unknown")
        return "; ".join(parts)

    interp_lines = []
    # OLS interpretation
    if results_object["ols_clustered"] is not None:
        o_beauty = results_object["ols_clustered"].get("beauty", {})
        o_beauty_sq = results_object["ols_clustered"].get("beauty_sq", {})
        interp_lines.append("OLS (clustered SEs):")
        interp_lines.append("  beauty: " + _interpret(o_beauty))
        interp_lines.append("  beauty_sq: " + _interpret(o_beauty_sq))

        # Try to give a short substantive interpretation if both coefs present
        try:
            cb = o_beauty.get("coef")
            cs = o_beauty_sq.get("coef")
            pb = o_beauty.get("pvalue")
            ps = o_beauty_sq.get("pvalue")
            if cb is not None and cs is not None:
                # determine shape
                if cs < 0:
                    shape = "concave (increases at low beauty then levels/decreases at high beauty)"
                elif cs > 0:
                    shape = "convex (accelerating effect at higher beauty)"
                else:
                    shape = "approximately linear"
                sig_text = ""
                if (pb is not None and pb < 0.05) or (ps is not None and ps < 0.05):
                    sig_text = " At least one term is statistically significant."
                interp_lines.append(f"  Substantive: Linear coef {cb:.4f}, quadratic coef {cs:.4f} -> {shape}.{sig_text}")
        except Exception:
            pass

    # Mixed model interpretation
    if results_object["mixedlm"] is not None:
        m_beauty = results_object["mixedlm"].get("beauty", {})
        m_beauty_sq = results_object["mixedlm"].get("beauty_sq", {})
        interp_lines.append("Mixed effects (random intercept for instructor):")
        interp_lines.append("  beauty: " + _interpret(m_beauty))
        interp_lines.append("  beauty_sq: " + _interpret(m_beauty_sq))

        try:
            cb = m_beauty.get("coef")
            cs = m_beauty_sq.get("coef")
            pb = m_beauty.get("pvalue")
            ps = m_beauty_sq.get("pvalue")
            if cb is not None and cs is not None:
                if cs < 0:
                    shape = "concave (increases at low beauty then levels/decreases at high beauty)"
                elif cs > 0:
                    shape = "convex (accelerating effect at higher beauty)"
                else:
                    shape = "approximately linear"
                sig_text = ""
                if (pb is not None and pb < 0.05) or (ps is not None and ps < 0.05):
                    sig_text = " At least one term is statistically significant."
                interp_lines.append(f"  Substantive: Linear coef {cb:.4f}, quadratic coef {cs:.4f} -> {shape}.{sig_text}")
        except Exception:
            pass

    description = " | ".join(interp_lines) if interp_lines else "No model results available to interpret."

    return {"object": results_object, "description": description}