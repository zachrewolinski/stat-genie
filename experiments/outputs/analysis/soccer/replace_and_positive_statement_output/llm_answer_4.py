def extract_final_answer(model_output):
    """
    Extracts the coefficient, IRR, confidence interval, p-value, and a brief conclusion
    about whether dark-skinned players receive more red cards than light-skinned players.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, IRR, CI, p-value, n_obs, coef_name)
      - "description": brief textual interpretation in the context of the task
    """
    import numpy as np
    import math

    # Defensive retrieval of model/result objects and summaries
    model = model_output.get('model', None)
    params = model_output.get('params', None)
    conf_int_df = model_output.get('conf_int', None)
    irr_series = model_output.get('IRR', None)
    irr_ci_low = model_output.get('IRR_ci_lower', None)
    irr_ci_high = model_output.get('IRR_ci_upper', None)
    n_obs = model_output.get('n_obs', None)

    # Find the coefficient name for the Dark vs Light contrast.
    # Preferred exact name used in the model code:
    preferred_name = 'C(SkinToneBin, Treatment("Light"))[T.Dark]'
    coef_name = None
    if params is not None and preferred_name in params.index:
        coef_name = preferred_name
    else:
        # fallback: try to find any parameter that mentions SkinTone and Dark (or just SkinTone)
        if params is not None:
            for name in params.index:
                lname = name.lower()
                if 'skintone' in lname and 'dark' in lname:
                    coef_name = name
                    break
            if coef_name is None:
                # maybe only "Dark" appears
                for name in params.index:
                    if 'dark' in name.lower():
                        coef_name = name
                        break
            if coef_name is None:
                # final fallback: any parameter mentioning "skintone"
                for name in params.index:
                    if 'skintone' in name.lower():
                        coef_name = name
                        break

    # Prepare outputs, default to None if we can't find them
    coef = None
    irr = None
    ci_lower = None
    ci_upper = None
    pvalue = None

    try:
        if coef_name is not None and params is not None and coef_name in params.index:
            coef = float(params[coef_name])
        elif model is not None:
            # try to get from model.params
            mp = getattr(model, 'params', None)
            if mp is not None:
                # choose the same heuristic for name
                for name in mp.index:
                    if coef_name is None or name == coef_name:
                        if 'skintone' in name.lower() or 'dark' in name.lower():
                            coef_name = name
                            coef = float(mp[name])
                            break
    except Exception:
        coef = None

    # IRR: prefer precomputed series if available, else compute from coef
    try:
        if irr_series is not None and coef_name in irr_series.index:
            irr = float(irr_series[coef_name])
        elif coef is not None:
            irr = float(np.exp(coef))
    except Exception:
        irr = None

    # Confidence intervals for the log-coef, then exponentiate to get IRR CI
    try:
        if conf_int_df is not None and coef_name in conf_int_df.index:
            lo_log = float(conf_int_df.loc[coef_name, 0])
            hi_log = float(conf_int_df.loc[coef_name, 1])
            ci_lower = float(math.exp(lo_log))
            ci_upper = float(math.exp(hi_log))
        elif model is not None:
            # try model.conf_int()
            confm = getattr(model, 'conf_int', None)
            if callable(confm):
                conf_df = model.conf_int()
            else:
                conf_df = confm
            if conf_df is not None and coef_name in conf_df.index:
                lo_log = float(conf_df.loc[coef_name, 0])
                hi_log = float(conf_df.loc[coef_name, 1])
                ci_lower = float(math.exp(lo_log))
                ci_upper = float(math.exp(hi_log))
    except Exception:
        ci_lower = ci_lower or None
        ci_upper = ci_upper or None

    # p-value: try to read from model.pvalues (should reflect robust/clustered p-values if available)
    try:
        if model is not None:
            pvals = getattr(model, 'pvalues', None)
            if pvals is not None:
                # if coef_name resolved, use it; otherwise try same heuristics
                if coef_name is not None and coef_name in pvals.index:
                    pvalue = float(pvals[coef_name])
                else:
                    # search for a matching p-value index
                    for name in pvals.index:
                        if 'skintone' in name.lower() or 'dark' in name.lower():
                            coef_name = name
                            pvalue = float(pvals[name])
                            break
    except Exception:
        pvalue = None

    # If n_obs not provided, try to get from model
    try:
        if n_obs is None and model is not None:
            n_obs = int(getattr(model, 'nobs', None))
    except Exception:
        pass

    # Build a concise conclusion about whether dark-skinned players are more likely
    conclusion = None
    if coef is None and irr is None:
        conclusion = ("Could not find a SkinToneBin (Dark vs Light) coefficient in the model output. "
                      "No conclusion can be drawn.")
    else:
        # Interpret direction and significance
        # If p-value is available, use it to judge statistical significance at alpha=0.05
        sig = None
        if pvalue is not None:
            sig = (pvalue < 0.05)
        # If p-value not available but CI available, check whether CI includes 1
        if sig is None and (ci_lower is not None and ci_upper is not None):
            sig = not (ci_lower <= 1.0 <= ci_upper)

        # Form conclusion text
        if sig is True:
            if coef is not None and coef > 0:
                conclusion = "Yes — statistically significant positive association: dark-skinned players receive more red cards."
            elif coef is not None and coef < 0:
                conclusion = "No — statistically significant negative association: dark-skinned players receive fewer red cards."
            else:
                conclusion = "Statistically significant effect found (direction unclear)."
        else:
            # Not significant / inconclusive
            conclusion = (
                "No evidence of a statistically significant difference in red card rates between "
                "dark-skinned and light-skinned players. "
            )
            # Add observed point-estimate info if available
            parts = []
            if irr is not None:
                parts.append(f"Estimated IRR = {irr:.3f}")
            if ci_lower is not None and ci_upper is not None:
                parts.append(f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
            if pvalue is not None:
                parts.append(f"p = {pvalue:.3f}")
            if parts:
                conclusion += " (" + "; ".join(parts) + ")."

    # Pack numeric object to return
    numeric_object = {
        'coef_name': coef_name,
        'coef_log_IRR': None if coef is None else float(coef),
        'IRR': None if irr is None else float(irr),
        'IRR_95CI_lower': None if ci_lower is None else float(ci_lower),
        'IRR_95CI_upper': None if ci_upper is None else float(ci_upper),
        'p_value': None if pvalue is None else float(pvalue),
        'n_obs': None if n_obs is None else int(n_obs)
    }

    description = (
        "Primary test: Are dark-skinned players more likely than light-skinned players to receive red cards?\n"
        f"Extracted coefficient: {numeric_object['coef_name']}\n"
        f"Log-IRR = {numeric_object['coef_log_IRR']}, IRR = {numeric_object['IRR']}, "
        f"95% CI = [{numeric_object['IRR_95CI_lower']}, {numeric_object['IRR_95CI_upper']}], "
        f"p-value = {numeric_object['p_value']}. "
        f"Number of dyad observations = {numeric_object['n_obs']}.\n\n"
        f"Interpretation: {conclusion}"
    )

    return {"object": numeric_object, "description": description}