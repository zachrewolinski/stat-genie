def extract_final_answer(model_output):
    """
    Extracts statistics about the MasFem (masculinity-femininity) coefficient from the
    model_output produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, p-values, 95% CIs,
                  and effect-size transformations (IRR / percent change) for both
                  the negative binomial and OLS (log-deaths) models when available.
      - "description": a short interpretation of those statistics relative to the
                       hypothesis "more feminine names -> fewer deaths".

    The function handles the case where model_output is a dict-like object with keys
    'negative_binomial' and 'ols_log_deaths' (as in the modeling code).
    """
    import numpy as np

    # Helper to safely extract stats for a given results object and parameter name
    def _extract_stats(res, param_name):
        out = {"present": False}
        try:
            params = getattr(res, "params")
            if param_name not in params.index:
                # Some wrappers expose params as ndarray with index in res.model.exog_names
                try:
                    exog_names = getattr(res.model, "exog_names", None)
                    if exog_names and param_name in exog_names:
                        # find position
                        idx = exog_names.index(param_name)
                        coef = float(params[idx])
                    else:
                        return out
                except Exception:
                    return out
            else:
                coef = float(params[param_name])
            # standard errors and p-values
            try:
                bse = float(res.bse[param_name]) if param_name in getattr(res, "bse").index else float(res.bse[exog_names.index(param_name)])
            except Exception:
                # fallback: try scalar bse (unlikely) or nan
                try:
                    bse = float(res.bse)
                except Exception:
                    bse = np.nan
            try:
                pval = float(res.pvalues[param_name]) if param_name in getattr(res, "pvalues").index else float(res.pvalues[exog_names.index(param_name)])
            except Exception:
                pval = np.nan
            # confidence intervals
            try:
                ci = res.conf_int().loc[param_name].astype(float).tolist()
            except Exception:
                try:
                    # maybe conf_int returns ndarray in same order as exog_names
                    ci_arr = res.conf_int()
                    if hasattr(res.model, "exog_names") and param_name in res.model.exog_names:
                        idx = res.model.exog_names.index(param_name)
                        ci = [float(ci_arr[idx, 0]), float(ci_arr[idx, 1])]
                    else:
                        ci = [np.nan, np.nan]
                except Exception:
                    ci = [np.nan, np.nan]

            out.update({
                "present": True,
                "coef": coef,
                "std_err": bse,
                "p_value": pval,
                "ci_lower": float(ci[0]) if ci is not None else np.nan,
                "ci_upper": float(ci[1]) if ci is not None else np.nan
            })
            return out
        except Exception:
            return out

    result = {"object": {}, "description": ""}

    # Determine how model_output is structured
    nb_res = None
    ols_res = None
    if isinstance(model_output, dict):
        nb_res = model_output.get("negative_binomial", None)
        ols_res = model_output.get("ols_log_deaths", None)
    else:
        # If a single results object was passed, try to infer which it is (prefer NB-like)
        nb_res = getattr(model_output, "negative_binomial", None)
        ols_res = getattr(model_output, "ols_log_deaths", None)
        if nb_res is None and hasattr(model_output, "model"):
            # assume this single object is the negative binomial
            nb_res = model_output

    # Extract stats for MasFem from each model if present
    analysis = {}
    for label, res in (("negative_binomial", nb_res), ("ols_log_deaths", ols_res)):
        if res is None:
            analysis[label] = {"available": False}
            continue
        stats = _extract_stats(res, "MasFem")
        if not stats.get("present", False):
            analysis[label] = {"available": False}
            continue
        # compute transformed effect sizes
        coef = stats["coef"]
        try:
            irr = float(np.exp(coef))  # incidence rate ratio for count model
        except Exception:
            irr = np.nan
        try:
            # percent change interpretation (approx): (exp(coef)-1)*100
            pct_change = (np.exp(coef) - 1.0) * 100.0
        except Exception:
            pct_change = np.nan

        stats.update({"IRR_or_multiplicative": irr, "percent_change": pct_change})
        analysis[label] = {"available": True, "stats": stats}

    result["object"]["analysis"] = analysis

    # Build interpretation logic
    nb_info = analysis.get("negative_binomial", {})
    ols_info = analysis.get("ols_log_deaths", {})

    interpretations = []
    # Check NB first
    if nb_info.get("available", False):
        s = nb_info["stats"]
        coef = s["coef"]
        p = s["p_value"]
        irr = s["IRR_or_multiplicative"]
        pct = s["percent_change"]
        interpretations.append(
            f"Negative binomial: MasFem coef = {coef:.4f}, p = {p:.4g}; "
            f"IRR = {irr:.4f} (expected multiplicative change per 1-unit increase), "
            f"≈ {pct:.2f}% change in expected deaths."
        )
    else:
        interpretations.append("Negative binomial: MasFem not available in model output.")

    # OLS on log-deaths
    if ols_info.get("available", False):
        s = ols_info["stats"]
        coef = s["coef"]
        p = s["p_value"]
        pct = s["percent_change"]
        interpretations.append(
            f"OLS (log(Deaths+1)): MasFem coef = {coef:.4f}, p = {p:.4g}; "
            f"Approx. {pct:.2f}% change in (Deaths+1) per 1-unit increase in MasFem."
        )
    else:
        interpretations.append("OLS (log-deaths): MasFem not available in model output.")

    # Decide whether evidence supports the hypothesis (more feminine -> fewer deaths)
    conclusion = "Inconclusive."
    # Use sign and statistical significance rules
    nb_sign = None
    ols_sign = None
    if nb_info.get("available", False):
        nb_coef = nb_info["stats"]["coef"]
        nb_p = nb_info["stats"]["p_value"]
        nb_sign = {"coef": nb_coef, "p": nb_p, "negative": nb_coef < 0}
    if ols_info.get("available", False):
        ols_coef = ols_info["stats"]["coef"]
        ols_p = ols_info["stats"]["p_value"]
        ols_sign = {"coef": ols_coef, "p": ols_p, "negative": ols_coef < 0}

    # Apply logic:
    # Strong support if NB negative and p<0.05
    if nb_sign and nb_sign["negative"] and nb_sign["p"] < 0.05:
        conclusion = "Supported: Negative binomial model shows a statistically significant negative association (higher femininity -> fewer deaths)."
    # If NB not significant but OLS significant negative
    elif (nb_sign is None or not (nb_sign["negative"] and nb_sign["p"] < 0.05)) and ols_sign and ols_sign["negative"] and ols_sign["p"] < 0.05:
        conclusion = "Partially supported: OLS on log-deaths shows a statistically significant negative association, but the negative binomial model does not reach conventional significance."
    # If both negative but neither significant
    elif (nb_sign and nb_sign["negative"] or nb_sign is None) and (ols_sign and ols_sign["negative"] or ols_sign is None):
        # both negative or one missing, but not significant
        ns_count = 0
        sig_count = 0
        if nb_sign:
            sig_count += int(nb_sign["p"] < 0.05)
            ns_count += int(nb_sign["p"] >= 0.05)
        if ols_sign:
            sig_count += int(ols_sign["p"] < 0.05)
            ns_count += int(ols_sign["p"] >= 0.05)
        if sig_count == 0:
            conclusion = "Directionally consistent (both coefficients negative) but not statistically significant in available models."
    else:
        # Conflicting signs or evidence against hypothesis
        conflict = False
        if nb_sign and ols_sign:
            if nb_sign["negative"] != ols_sign["negative"]:
                conflict = True
        if conflict:
            conclusion = "Conflicting evidence: models produce coefficients with opposite signs; no consistent support for the hypothesis."
        else:
            # neither supports
            conclusion = "No evidence supporting the hypothesis in the available models."

    result["description"] = " | ".join(interpretations) + " || Conclusion: " + conclusion

    return result