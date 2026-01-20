def extract_final_answer(model_output):
    """
    Extracts statistics for the 'children_binary' variable from a fitted model_output dict
    containing 'zinb_results' and 'ols_results' (statsmodels result wrappers).
    Returns a dict with keys:
      - "object": dict of extracted numeric results (ZINB count part primary, ZINB inflation part,
                  and OLS robustness check).
      - "description": plain-language interpretation of the effect of having children on affairs.
    """
    import numpy as np

    res = {}

    # Helper to safely extract param, se, pvalue, ci for a named parameter
    def extract_from_result(result, name):
        out = {}
        params = result.params  # pandas Series
        # If the requested name is not present, return None
        if name not in params.index:
            return None
        coef = float(params.loc[name])
        # standard error
        try:
            se = float(result.bse.loc[name])
        except Exception:
            # fallback if bse is array-like
            se = float(result.bse[list(params.index).index(name)])
        # p-value (z/t depending on model)
        try:
            pval = float(result.pvalues.loc[name])
        except Exception:
            pval = float(result.pvalues[list(params.index).index(name)])
        # confidence interval
        try:
            ci_all = result.conf_int()
            # ci_all may be DataFrame or ndarray; align by parameter position
            if hasattr(ci_all, "loc") and name in getattr(ci_all, "index", []):
                ci_row = ci_all.loc[name].values
            else:
                pos = list(params.index).index(name)
                ci_row = np.asarray(ci_all)[pos]
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        except Exception:
            # fallback: +/- 1.96*se
            ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se

        out.update({
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        return out

    # Extract from ZINB results
    zinb = model_output.get('zinb_results', None)
    if zinb is None:
        raise ValueError("model_output must contain 'zinb_results'")

    # Identify count and inflation parameter names from the model object, if available
    try:
        count_names = list(zinb.model.exog_names)
    except Exception:
        # fallback: take first block of params until 'inflate_' or 'alpha' appears
        count_names = list(zinb.params.index)  # best effort; extraction function will check presence
    try:
        infl_names = list(zinb.model.exog_infl_names)
    except Exception:
        # fallback: find params with 'inflate' prefix if any
        infl_names = [n for n in zinb.params.index if n.startswith('inflate')]

    # Preferred: extract using model.exog_names for correct mapping
    # Get count-part parameter for children_binary
    children_name = 'children_binary'
    zinb_count_res = extract_from_result(zinb, children_name)
    # Also get inflation-part parameter if present (name might appear as same or prefixed)
    # The inflation names may be identical to exog names but stored with 'inflate_' prefix in params.
    inflation_param_name = None
    # Try typical naming conventions
    possible_infl_names = [
        'inflate_' + children_name,
        children_name + '_infl',
        children_name  # sometimes exog_infl uses same name but appears after count params
    ]
    # Check which of these actually present in params
    for nm in possible_infl_names:
        if nm in zinb.params.index:
            inflation_param_name = nm
            break
    # If not found but model.exog_infl_names exists and contains children_name, find its exact param name in params
    if inflation_param_name is None:
        try:
            if hasattr(zinb.model, 'exog_infl_names') and children_name in zinb.model.exog_infl_names:
                # find the param name by matching order: after count params in params.index
                # We'll attempt to find any param that contains children_name and is not the count one
                for nm in zinb.params.index:
                    if children_name in nm and nm != children_name:
                        inflation_param_name = nm
                        break
        except Exception:
            pass

    zinb_infl_res = None
    if inflation_param_name:
        zinb_infl_res = extract_from_result(zinb, inflation_param_name)

    # For count-part, compute IRR = exp(coef) and IRR CI
    if zinb_count_res is not None:
        irr = float(np.exp(zinb_count_res['coef']))
        irr_ci_lower = float(np.exp(zinb_count_res['ci_lower']))
        irr_ci_upper = float(np.exp(zinb_count_res['ci_upper']))
        zinb_count_res.update({
            "irr": irr,
            "irr_ci_lower": irr_ci_lower,
            "irr_ci_upper": irr_ci_upper,
            "significant": zinb_count_res['pvalue'] < 0.05
        })
    else:
        zinb_count_res = {"error": f"Parameter '{children_name}' not found in ZINB count part."}

    if zinb_infl_res is not None:
        # For the inflation logit part, exp(coef) is the odds ratio for being a structural zero
        infl_or = float(np.exp(zinb_infl_res['coef']))
        infl_or_ci_lower = float(np.exp(zinb_infl_res['ci_lower']))
        infl_or_ci_upper = float(np.exp(zinb_infl_res['ci_upper']))
        zinb_infl_res.update({
            "odds_ratio": infl_or,
            "odds_ratio_ci_lower": infl_or_ci_lower,
            "odds_ratio_ci_upper": infl_or_ci_upper,
            "significant": zinb_infl_res['pvalue'] < 0.05
        })

    # Extract from OLS results for robustness
    ols = model_output.get('ols_results', None)
    ols_res = None
    if ols is not None:
        ols_children = extract_from_result(ols, children_name)
        if ols_children is not None:
            # OLS interpretation: additive change in expected count
            ols_children.update({"significant": ols_children['pvalue'] < 0.05})
            ols_res = ols_children
        else:
            ols_res = {"error": f"Parameter '{children_name}' not found in OLS model."}

    # Prepare the object to return
    res_object = {
        "zinb_count_children": zinb_count_res,
        "zinb_inflation_children": zinb_infl_res,
        "ols_children": ols_res
    }

    # Brief interpretation:
    # Focus on the ZINB count-part IRR: IRR < 1 means having children is associated with fewer expected affairs;
    # IRR > 1 means associated with more expected affairs. Use p-value to assess statistical significance.
    interp_lines = []
    if isinstance(zinb_count_res, dict) and "error" not in zinb_count_res:
        coef = zinb_count_res["coef"]
        p = zinb_count_res["pvalue"]
        irr = zinb_count_res["irr"]
        ci = (zinb_count_res["ci_lower"], zinb_count_res["ci_upper"])
        irr_ci = (zinb_count_res["irr_ci_lower"], zinb_count_res["irr_ci_upper"])
        sig_text = "statistically significant (p < 0.05)" if zinb_count_res["significant"] else "not statistically significant (p >= 0.05)"
        direction = "decrease" if irr < 1 else ("increase" if irr > 1 else "no change")
        interp_lines.append(
            f"ZINB (count part): children_binary coef = {coef:.4f}, p = {p:.4g}; "
            f"IRR = {irr:.4f} (95% CI {irr_ci[0]:.4f} to {irr_ci[1]:.4f}). "
            f"This indicates a {direction} in the expected number of extramarital affairs associated with having children; the effect is {sig_text}."
        )
    else:
        interp_lines.append("ZINB count-part statistic for 'children_binary' could not be located.")

    if zinb_infl_res is not None and "error" not in zinb_infl_res:
        coef = zinb_infl_res["coef"]
        p = zinb_infl_res["pvalue"]
        orv = zinb_infl_res["odds_ratio"]
        or_ci = (zinb_infl_res["odds_ratio_ci_lower"], zinb_infl_res["odds_ratio_ci_upper"])
        sig_text = "statistically significant (p < 0.05)" if zinb_infl_res["significant"] else "not statistically significant (p >= 0.05)"
        interp_lines.append(
            f"ZINB (inflation part): inflation coef = {coef:.4f}, p = {p:.4g}; "
            f"odds ratio for being a structural-zero = {orv:.4f} (95% CI {or_ci[0]:.4f} to {or_ci[1]:.4f}). "
            f"A value >1 implies higher odds of being an 'excess zero' (structural non-participant). Effect is {sig_text}."
        )

    if ols_res is not None:
        if "error" not in ols_res:
            coef = ols_res["coef"]
            p = ols_res["pvalue"]
            ci = (ols_res["ci_lower"], ols_res["ci_upper"])
            sig_text = "statistically significant (p < 0.05)" if ols_res["significant"] else "not statistically significant (p >= 0.05)"
            interp_lines.append(
                f"OLS (robustness): children_binary coef = {coef:.4f}, p = {p:.4g}; "
                f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]. Interpretation: having children is associated with an average change of {coef:.4f} affairs per year. Effect is {sig_text}."
            )
        else:
            interp_lines.append("OLS statistic for 'children_binary' could not be located.")

    description = " ".join(interp_lines)

    return {"object": res_object, "description": description}