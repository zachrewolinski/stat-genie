def extract_final_answer(model_output):
    """
    Extract relevant statistics about the effect of name femininity from the modeling output.
    Returns a dictionary with keys:
      - "object": dict with per-model extracted info (or errors)
      - "description": human-readable interpretation and, if models failed, a suggested fix.
    """
    import numpy as np
    summary = {}
    # Expected model keys we attempted to run
    keys = ['nb_alldeaths', 'ols_log_alldeaths', 'ols_log_ndam15', 'nb_alldeaths_mturk_iv']

    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict. Cannot extract results."
        }

    for key in keys:
        val = model_output.get(key, None)
        if val is None:
            summary[key] = {"status": "missing", "note": "Model not present in output."}
            continue

        # If the value is a string, it's an error message returned by the modeling function
        if isinstance(val, str):
            summary[key] = {"status": "error", "error_message": val}
            continue

        # Otherwise, try to extract stats from a statsmodels-like fitted result object
        try:
            # determine parameter name for femininity
            param_name = 'masfem_mturk_scaled' if key == 'nb_alldeaths_mturk_iv' else 'masfem_scaled'

            # Access attributes commonly available on statsmodels results
            params = getattr(val, "params", None)
            pvalues = getattr(val, "pvalues", None)
            bse = getattr(val, "bse", None)
            conf_int = None
            try:
                conf_int = val.conf_int()
            except Exception:
                # some result objects name the method differently or require args; ignore if unavailable
                conf_int = None

            if params is None or param_name not in params:
                # try alternative access (e.g., params is an array)
                summary[key] = {"status": "extraction_error", "error_message": f"Parameter '{param_name}' not found in model.params."}
                continue

            coef = float(params[param_name])
            pval = float(pvalues[param_name]) if (pvalues is not None and param_name in pvalues) else None
            se = float(bse[param_name]) if (bse is not None and param_name in bse) else None
            ci = None
            if conf_int is not None:
                try:
                    # conf_int may be a DataFrame or 2D array; try indexing by param_name
                    if hasattr(conf_int, "loc"):
                        low, high = conf_int.loc[param_name].tolist()
                    else:
                        # assume order of params matches
                        idx = list(params.index).index(param_name) if hasattr(params, "index") else None
                        if idx is not None and idx < len(conf_int):
                            low, high = float(conf_int[idx, 0]), float(conf_int[idx, 1])
                        else:
                            low, high = None, None
                    ci = [low, high]
                except Exception:
                    ci = None

            # For count model or logged outcome, exponentiated coefficient is often informative
            try:
                exp_coef = float(np.exp(coef))
                exp_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))] if (ci is not None and ci[0] is not None and ci[1] is not None) else None
            except Exception:
                exp_coef = None
                exp_ci = None

            # significance judgement (conservative): p < 0.05
            significance = None
            if pval is not None:
                significance = (pval < 0.05)

            summary[key] = {
                "status": "ok",
                "param_name": param_name,
                "coef": coef,
                "std_err": se,
                "pvalue": pval,
                "ci_95": ci,
                "exp_coef": exp_coef,
                "exp_ci_95": exp_ci,
                "significant_at_0.05": significance
            }
        except Exception as e:
            summary[key] = {"status": "extraction_error", "error_message": str(e)}

    # Build a human-readable description
    # If all models errored, provide an explanation and a suggested fix
    all_error_like = all(
        (summary[k]["status"] in ("error", "missing", "extraction_error"))
        for k in summary
    )

    if all_error_like:
        # collect unique error messages
        error_msgs = []
        for k, v in summary.items():
            if v.get("status") == "error":
                error_msgs.append(f"{k}: {v.get('error_message')}")
            elif v.get("status") == "extraction_error":
                error_msgs.append(f"{k}: {v.get('error_message')}")
            elif v.get("status") == "missing":
                error_msgs.append(f"{k}: missing")

        # Common cause in the provided run: "Cannot interpret 'Int64Dtype()' as a data type"
        suggested_fix = (
            "All models failed to run (see errors). A common cause is pandas' nullable integer dtype "
            "(Int64) being incompatible with statsmodels. Suggested fixes before rerunning models:\n"
            " - Convert nullable integer dtypes to numpy dtypes, e.g.:\n"
            "     df['category'] = df['category'].astype('int64')\n"
            "     df['alldeaths'] = df['alldeaths'].astype('float64')\n"
            " - Or coerce all modeling columns to numeric numpy types:\n"
            "     model_cols = ['alldeaths','masfem_scaled','wind_scaled','min_scaled','category','elapsedyrs','gender_mf']\n"
            "     df[model_cols] = df[model_cols].apply(pd.to_numeric, errors='coerce').astype(float)\n"
            "After converting dtypes, rerun the models and then re-call this extraction function on the new output."
        )

        description = (
            "No model coefficients could be extracted because all model runs returned errors. "
            "Errors: " + "; ".join(error_msgs) + ". " + suggested_fix
        )

        return {"object": summary, "description": description}

    # If some models succeeded, summarize findings for the primary model(s)
    parts = []
    primary = summary.get('nb_alldeaths')
    if primary and primary.get('status') == 'ok':
        coef = primary['coef']
        pval = primary['pvalue']
        sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig_text = ("statistically significant (p={:.3g})".format(pval) if pval is not None and primary['significant_at_0.05']
                    else ("not statistically significant (p={:.3g})".format(pval) if pval is not None else "p-value unavailable"))
        irr = primary.get('exp_coef')
        irr_text = f"IRR = {irr:.3g}. " if irr is not None else ""
        parts.append(f"Primary NB model (alldeaths ~ masfem_scaled): coefficient on masfem_scaled = {coef:.4g} ({sign}), {sig_text}. {irr_text}")

    # Add OLS robustness if present
    ols = summary.get('ols_log_alldeaths')
    if ols and ols.get('status') == 'ok':
        coef = ols['coef']
        pval = ols['pvalue']
        sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig_text = ("statistically significant (p={:.3g})".format(pval) if pval is not None and ols['significant_at_0.05']
                    else ("not statistically significant (p={:.3g})".format(pval) if pval is not None else "p-value unavailable"))
        parts.append(f"OLS on log deaths: coefficient on masfem_scaled = {coef:.4g} ({sign}), {sig_text}.")

    # Add damage model if present
    dam = summary.get('ols_log_ndam15')
    if dam and dam.get('status') == 'ok':
        coef = dam['coef']
        pval = dam['pvalue']
        sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig_text = ("statistically significant (p={:.3g})".format(pval) if pval is not None and dam['significant_at_0.05']
                    else ("not statistically significant (p={:.3g})".format(pval) if pval is not None else "p-value unavailable"))
        parts.append(f"OLS on log economic damage: coefficient on masfem_scaled = {coef:.4g} ({sign}), {sig_text}.")

    alt = summary.get('nb_alldeaths_mturk_iv')
    if alt and alt.get('status') == 'ok':
        coef = alt['coef']
        pval = alt['pvalue']
        sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig_text = ("statistically significant (p={:.3g})".format(pval) if pval is not None and alt['significant_at_0.05']
                    else ("not statistically significant (p={:.3g})".format(pval) if pval is not None else "p-value unavailable"))
        irr = alt.get('exp_coef')
        irr_text = f"IRR = {irr:.3g}. " if irr is not None else ""
        parts.append(f"Alternative IV NB model (masfem_mturk_scaled): coefficient = {coef:.4g} ({sign}), {sig_text}. {irr_text}")

    # Compose final interpretation
    if parts:
        conclusion = " ".join(parts)
        # Quick inference: if any significant positive coefficient, that supports the hypothesis (feminine names -> more deaths)
        pos_sig = any((v.get('status') == 'ok' and v.get('significant_at_0.05') and v.get('coef') > 0) for v in summary.values())
        neg_sig = any((v.get('status') == 'ok' and v.get('significant_at_0.05') and v.get('coef') < 0) for v in summary.values())
        if pos_sig and not neg_sig:
            overall = "The (available) model results provide evidence consistent with the hypothesis: more feminine hurricane names are associated with greater adverse outcomes (consistent with fewer precautions)."
        elif neg_sig and not pos_sig:
            overall = "The (available) model results provide evidence contrary to the hypothesis: more feminine hurricane names are associated with fewer adverse outcomes."
        elif pos_sig and neg_sig:
            overall = "Different models show significant effects in opposite directions; results are mixed."
        else:
            overall = "No statistically significant effect of name femininity was found in the available models."
        description = overall + " Details: " + conclusion
    else:
        # No successful extraction but not all errored (edge case)
        description = "No model results could be extracted successfully; see 'object' for per-model statuses and errors."

    return {"object": summary, "description": description}