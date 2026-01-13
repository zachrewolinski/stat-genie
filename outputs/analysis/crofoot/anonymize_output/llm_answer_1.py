def extract_final_answer(model_output):
    """
    Extracts key statistics for the primary predictors (relative group size,
    location advantage, and their interaction) from the model_output returned
    by the modeling function.

    Returns a dictionary with:
      - "object": dict mapping each target predictor to its extracted stats
                  (coefficient, SE, p-value, 95% CI on log-odds scale,
                   odds ratio and 95% CI on OR scale).
                  If the model is not available, this will be None.
      - "description": brief interpretation of what the returned numbers mean
                       and which model/results were used.
    """
    import numpy as np

    # Defensive checks
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict. Cannot extract statistics."
        }

    # Try to use clustered_result if present; otherwise glm_result
    res = model_output.get('clustered_result') or model_output.get('glm_result')
    n_obs = model_output.get('n_obs', None)
    formula = model_output.get('formula', None)

    if res is None:
        desc = "No fitted model available in model_output (both 'clustered_result' and 'glm_result' are None)."
        if n_obs is not None:
            desc += f" n_obs reported = {n_obs}."
        if formula:
            desc += f" Expected formula: {formula}."
        return {"object": None, "description": desc}

    # Helper to find parameter name for interaction regardless of ordering
    def find_param_name(params_index, name_a, name_b):
        # Exact possibilities: "name_a:name_b" or "name_b:name_a"
        cand1 = f"{name_a}:{name_b}"
        cand2 = f"{name_b}:{name_a}"
        if cand1 in params_index:
            return cand1
        if cand2 in params_index:
            return cand2
        # as fallback, find any param that contains both substrings
        for p in params_index:
            if (name_a in p) and (name_b in p):
                return p
        return None

    params = None
    pvalues = None
    bse = None
    conf = None
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        # conf_int may require calling as method
        try:
            conf = res.conf_int()
        except Exception:
            # some robustcov result objects still support conf_int
            conf = None
    except Exception as e:
        return {
            "object": None,
            "description": f"Model result object present but expected attributes missing: {e}"
        }

    # Names of primary predictors in the formula
    target_main1 = "RelSize_z"
    target_main2 = "LocAdv_z"
    target_inter = find_param_name(params.index, target_main1, target_main2)

    # Function to extract stats for a parameter name
    def extract_for_param(pname):
        if pname is None or pname not in params.index:
            return None
        coef = float(params.loc[pname])
        se = float(bse.loc[pname]) if (bse is not None and pname in bse.index) else None
        pval = float(pvalues.loc[pname]) if (pvalues is not None and pname in pvalues.index) else None
        # confidence interval on log-odds
        if conf is not None and pname in conf.index:
            lower_log = float(conf.loc[pname, 0])
            upper_log = float(conf.loc[pname, 1])
        else:
            # If conf_int not available, try to compute +/- 1.96*se when se present
            if se is not None:
                lower_log = coef - 1.96 * se
                upper_log = coef + 1.96 * se
            else:
                lower_log = upper_log = None
        # odds ratio and CI
        or_val = float(np.exp(coef)) if coef is not None else None
        or_ci_lower = float(np.exp(lower_log)) if lower_log is not None else None
        or_ci_upper = float(np.exp(upper_log)) if upper_log is not None else None

        return {
            "param_name": pname,
            "coef_logodds": coef,
            "se": se,
            "p_value": pval,
            "conf_int_logodds": [lower_log, upper_log],
            "odds_ratio": or_val,
            "odds_ratio_95CI": [or_ci_lower, or_ci_upper]
        }

    results = {
        "RelSize_z": extract_for_param(target_main1),
        "LocAdv_z": extract_for_param(target_main2),
        "Interaction_RelSize_x_LocAdv": extract_for_param(target_inter)
    }

    # Build a short interpretive description
    used_result_type = ("clustered_result" if model_output.get('clustered_result') is not None
                        else "glm_result")
    desc_lines = [
        f"Extracted coefficients and inferential statistics from the model result object ({used_result_type}).",
        f"Model formula: {formula}" if formula else "Model formula not provided.",
        f"Number of observations (n_obs) reported: {n_obs}" if n_obs is not None else "n_obs not provided."
    ]
    desc_lines.append("For each predictor, 'coef_logodds' is the estimated effect on the log-odds of the focal group winning;"
                      " 'odds_ratio' is exp(coef) (multiplicative change in odds per 1 SD increase);"
                      " 95% CIs for both log-odds and odds ratios are included where available.")
    # add note about p-values
    desc_lines.append("P-values are from the model result object provided (clustered robust SEs used if 'clustered_result' present).")

    # If any of the main predictors are missing, note that
    missing = [k for k, v in results.items() if v is None]
    if missing:
        desc_lines.append(f"Note: could not find parameters for: {', '.join(missing)}. Parameter naming in the fitted model may differ.")

    description = " ".join(desc_lines)

    return {"object": results, "description": description}