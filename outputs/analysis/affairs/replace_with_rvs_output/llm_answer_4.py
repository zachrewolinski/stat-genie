def extract_final_answer(model_output):
    """
    Extract statistics for the HasChildren variable from a fitted count model (ZINB or NB GLM).
    Returns a dictionary with keys:
      - "object": a dict with extracted numeric results (coefficients, p-values, CIs, IRR)
      - "description": a short interpretation of whether having children decreases extramarital affairs

    The function is defensive and attempts to handle:
      - statsmodels ZeroInflatedNegativeBinomialResultsWrapper (inflate_ prefix for inflation params)
      - statsmodels GLM/GLMResults (Negative Binomial fallback)
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to get a pandas Series of params, bse, pvalues and a DataFrame of conf_int
    def get_series(attr_name):
        attr = getattr(res, attr_name, None)
        if attr is None:
            return None
        # If it's already a Series with index, return as-is
        if isinstance(attr, pd.Series):
            return attr
        # If it's a numpy array but model has names, attempt to attach names
        try:
            arr = np.asarray(attr)
            names = None
            if hasattr(res, "params") and isinstance(res.params, pd.Series):
                names = res.params.index
            elif hasattr(res, "model") and hasattr(res.model, "exog_names"):
                # Try to build names for params; for ZINB these may not include inflation/alpha,
                # but it's a best-effort fallback.
                names = list(res.model.exog_names)
                if hasattr(res.model, "exog_infl_names"):
                    names += ['inflate_' + n for n in res.model.exog_infl_names]
            if names is not None and len(names) == len(arr):
                return pd.Series(arr, index=names)
        except Exception:
            pass
        # Last resort: return as a Series with integer index
        try:
            return pd.Series(attr)
        except Exception:
            return None

    params = get_series("params")
    bse = get_series("bse")
    pvalues = get_series("pvalues")

    # conf_int may be a DataFrame or array
    conf = None
    if hasattr(res, "conf_int"):
        try:
            conf_raw = res.conf_int()
            if isinstance(conf_raw, pd.DataFrame):
                conf = conf_raw
            else:
                # try create DataFrame if we can get names
                conf = pd.DataFrame(conf_raw, index=(params.index if params is not None else None))
        except Exception:
            conf = None

    # Find parameter names for HasChildren in count and inflation parts
    count_name = None
    infl_name = None
    if params is not None:
        names = list(params.index.astype(str))
        # count param: contains HasChildren but not 'inflate' in name
        for n in names:
            if "HasChildren" in n and "inflate" not in n.lower():
                count_name = n
                break
        # inflation param: contains HasChildren and 'inflate' in name (common convention: 'inflate_HasChildren')
        for n in names:
            if "HasChildren" in n and "inflate" in n.lower():
                infl_name = n
                break

    # If no explicit inflation param found, still check for names like 'HasChildren_infl' or similar
    if infl_name is None and params is not None:
        for n in names:
            if "HasChildren" in n and ("_infl" in n.lower() or ".infl" in n.lower() or "infl" in n.lower() and n != count_name):
                infl_name = n
                break

    result_obj = {}
    messages = []

    # Extract count-model statistics if present
    if count_name is not None and params is not None:
        coef = float(params[count_name])
        pv = float(pvalues[count_name]) if (pvalues is not None and count_name in pvalues.index) else None
        se = float(bse[count_name]) if (bse is not None and count_name in bse.index) else None
        ci_lower, ci_upper = (None, None)
        if conf is not None and count_name in conf.index:
            try:
                ci_lower, ci_upper = float(conf.loc[count_name].iloc[0]), float(conf.loc[count_name].iloc[1])
            except Exception:
                ci_lower, ci_upper = None, None
        # Incidence Rate Ratio (IRR) for the count model: exp(coef)
        irr = float(np.exp(coef))
        irr_ci = (np.exp(ci_lower) if ci_lower is not None else None, np.exp(ci_upper) if ci_upper is not None else None)

        result_obj['count'] = {
            'param_name': count_name,
            'coef (log count)': coef,
            'std_err': se,
            'p_value': pv,
            'conf_int_95': (ci_lower, ci_upper),
            'IRR': irr,
            'IRR_95_conf_int': irr_ci
        }

        # Interpretation based on IRR and p-value
        if pv is not None and pv < 0.05:
            if irr < 1:
                messages.append(
                    f"In the count part, HasChildren has a negative association with the expected number of affairs "
                    f"(coef = {coef:.4f}, IRR = {irr:.3f}, 95% CI IRR = [{irr_ci[0]:.3f}, {irr_ci[1]:.3f}], p = {pv:.3g}). "
                    "This indicates a statistically significant decrease in the expected count of extramarital affairs for respondents with children."
                )
            else:
                messages.append(
                    f"In the count part, HasChildren is associated with a higher expected number of affairs "
                    f"(coef = {coef:.4f}, IRR = {irr:.3f}, p = {pv:.3g})."
                )
        else:
            messages.append(
                f"In the count part, HasChildren coefficient = {coef:.4f} (IRR = {irr:.3f}), p = {pv if pv is not None else 'NA'}. "
                "This is not statistically significant at the 0.05 level, so we do not have strong evidence of an effect on the expected number of affairs."
            )
    else:
        messages.append("Could not find a count-model parameter for 'HasChildren' in the fitted model output.")

    # Extract inflation-model statistics (if present)
    if infl_name is not None and params is not None:
        icoef = float(params[infl_name])
        ipv = float(pvalues[infl_name]) if (pvalues is not None and infl_name in pvalues.index) else None
        ise = float(bse[infl_name]) if (bse is not None and infl_name in bse.index) else None
        ici_lower, ici_upper = (None, None)
        if conf is not None and infl_name in conf.index:
            try:
                ici_lower, ici_upper = float(conf.loc[infl_name].iloc[0]), float(conf.loc[infl_name].iloc[1])
            except Exception:
                ici_lower, ici_upper = None, None

        # For the inflation (logit) part, coef >0 means higher log-odds of being in the always-zero group
        # so a positive and significant inflation coef suggests children increase the probability of being always-zero (i.e., no affairs)
        infl_odds_ratio = float(np.exp(icoef))
        infl_or_ci = (np.exp(ici_lower) if ici_lower is not None else None, np.exp(ici_upper) if ici_upper is not None else None)

        result_obj['inflation'] = {
            'param_name': infl_name,
            'coef (logit inflation)': icoef,
            'std_err': ise,
            'p_value': ipv,
            'conf_int_95': (ici_lower, ici_upper),
            'odds_ratio': infl_odds_ratio,
            'odds_ratio_95_conf_int': infl_or_ci
        }

        if ipv is not None and ipv < 0.05:
            if icoef > 0:
                messages.append(
                    f"In the inflation part, HasChildren has a positive coefficient (coef = {icoef:.4f}, OR = {infl_odds_ratio:.3f}, p = {ipv:.3g}), "
                    "indicating that having children is associated with higher odds of being in the always-zero group (i.e., more likely to have zero affairs). "
                    "This supports the conclusion that children decrease the likelihood of any extramarital affairs."
                )
            else:
                messages.append(
                    f"In the inflation part, HasChildren has a negative coefficient (coef = {icoef:.4f}, OR = {infl_odds_ratio:.3f}, p = {ipv:.3g}), "
                    "indicating lower odds of being in the always-zero group (i.e., more likely to have any affairs)."
                )
        else:
            messages.append(
                f"In the inflation part, HasChildren coef = {icoef:.4f} (OR = {infl_odds_ratio:.3f}), p = {ipv if ipv is not None else 'NA'}. "
                "This is not statistically significant at the 0.05 level."
            )
    else:
        messages.append("No inflation-model parameter for 'HasChildren' was found (model may be a non-zero-inflated NB GLM).")

    # Summarize final conclusion combining count and inflation evidence
    conclusion = ""
    # Use available significance info to draw overall conclusion
    count_info = result_obj.get('count')
    infl_info = result_obj.get('inflation')

    sig_decrease_count = False
    sig_inflation_support = False
    sig_increase_count = False
    sig_inflation_against = False

    if count_info is not None and count_info['p_value'] is not None and count_info['p_value'] < 0.05:
        if count_info['IRR'] < 1:
            sig_decrease_count = True
        else:
            sig_increase_count = True

    if infl_info is not None and infl_info['p_value'] is not None and infl_info['p_value'] < 0.05:
        if infl_info['coef (logit inflation)'] > 0:
            sig_inflation_support = True
        else:
            sig_inflation_against = True

    if sig_decrease_count or sig_inflation_support:
        conclusion = "Overall: Evidence suggests that having children decreases engagement in extramarital affairs."
        # If both present, note both
        details = []
        if sig_decrease_count:
            details.append("significant negative association in the count model (lower expected number of affairs).")
        if sig_inflation_support:
            details.append("significant positive inflation effect (higher odds of being always-zero = no affairs).")
        if details:
            conclusion += " " + " Also: " + " ".join(details)
    elif sig_increase_count or sig_inflation_against:
        conclusion = "Overall: Evidence suggests having children is associated with higher engagement in affairs (contrary to the hypothesis)."
    else:
        conclusion = "Overall: No statistically significant evidence that having children decreases engagement in extramarital affairs (insufficient evidence)."

    # Attach model type name if possible
    model_type = type(res).__name__ if res is not None else "Unknown"
    result_obj['model_type'] = model_type
    result_obj['conclusion'] = conclusion
    result_obj['notes'] = messages

    # Provide a human-readable description summarizing key numbers and the conclusion
    description_lines = []
    if count_info is not None:
        description_lines.append(
            f"Count part: param '{count_info['param_name']}', coef (log count) = {count_info['coef (log count)']:.4f}, "
            f"IRR = {count_info['IRR']:.3f}, p = {count_info['p_value'] if count_info['p_value'] is not None else 'NA'}."
        )
    if infl_info is not None:
        description_lines.append(
            f"Inflation part: param '{infl_info['param_name']}', coef (logit) = {infl_info['coef (logit inflation)']:.4f}, "
            f"OR = {infl_info['odds_ratio']:.3f}, p = {infl_info['p_value'] if infl_info['p_value'] is not None else 'NA'}."
        )
    description_lines.append(conclusion)

    description = " ".join(description_lines)

    return {"object": result_obj, "description": description}