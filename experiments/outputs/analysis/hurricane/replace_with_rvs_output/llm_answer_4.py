def extract_final_answer(model_output):
    """
    Extracts coefficients, SEs, p-values, 95% CIs from statsmodels result objects
    returned in model_output, computes odds ratios for logit models, and
    produces a concise conclusion about whether more-feminine hurricane names
    are associated with higher human/economic costs (interpreted as fewer precautions).

    Returns:
      {
        "object": {
           "<model_key>": {
               "iv": <iv_name>,
               "coef": <float>,
               "se": <float>,
               "pvalue": <float>,
               "ci_lower": <float>,
               "ci_upper": <float>,
               // for logit models:
               "odds_ratio": <float>,
               "or_ci_lower": <float>,
               "or_ci_upper": <float>,
               "significant": <bool>
           },
           ...,
           "summary_by_iv": {
               "masfem_z": { ... aggregated counts ... },
               "gender_female": { ... },
               "masfem_mturk_z": { ... }
           },
           "final_judgement": "<Yes / No / Mixed> (rationale...)"
        },
        "description": "<human-readable interpretation>"
      }
    """
    import numpy as np
    import math

    summary = {}
    # helper to test if object is a statsmodels results instance
    def is_sm_res(obj):
        return hasattr(obj, "params") and hasattr(obj, "pvalues")

    # iterate through provided models
    for key, mod in (model_output or {}).items():
        entry = {
            "iv": None,
            "coef": None,
            "se": None,
            "pvalue": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant": None,
            "model_type": None
        }
        if mod is None:
            summary[key] = {"error": "model is None"}
            continue
        if not is_sm_res(mod):
            summary[key] = {"error": "not a statsmodels-like result object"}
            continue
        # Determine predictor: params index order is [const, predictor, controls...]
        try:
            param_index = list(mod.params.index)
            # if there's a const, predictor likely at index 1, else index 0
            if "const" in param_index:
                if len(param_index) >= 2:
                    iv = param_index[1]
                else:
                    iv = param_index[0]
            else:
                iv = param_index[0]
            entry["iv"] = iv
            coef = float(mod.params.loc[iv])
            entry["coef"] = coef
            # standard error and p-value
            try:
                se = float(mod.bse.loc[iv])
            except Exception:
                se = None
            entry["se"] = se
            try:
                pval = float(mod.pvalues.loc[iv])
            except Exception:
                pval = None
            entry["pvalue"] = pval
            # 95% CI
            try:
                ci = mod.conf_int(alpha=0.05)
                ci_lower = float(ci.loc[iv, 0])
                ci_upper = float(ci.loc[iv, 1])
            except Exception:
                # fallback if conf_int returns array-like with same order
                try:
                    ci_arr = mod.conf_int(alpha=0.05)
                    # find row index of iv
                    if hasattr(ci_arr, "index"):
                        ci_lower = float(ci_arr.loc[iv][0])
                        ci_upper = float(ci_arr.loc[iv][1])
                    else:
                        # assume order matches params
                        ix = param_index.index(iv)
                        ci_lower = float(ci_arr[ix, 0])
                        ci_upper = float(ci_arr[ix, 1])
                except Exception:
                    ci_lower = None
                    ci_upper = None
            entry["ci_lower"] = ci_lower
            entry["ci_upper"] = ci_upper
            # mark significance at conventional 0.05
            entry["significant"] = (pval is not None) and (pval < 0.05)
            # If logistic (discrete results), compute odds ratio and OR CI
            mod_class_name = mod.__class__.__name__.lower()
            if "binaryresultswrapper" in mod_class_name or "logit" in mod_class_name or "discrete" in mod_class_name:
                entry["model_type"] = "logit"
                try:
                    or_val = float(np.exp(coef))
                    or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
                    or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
                    entry["odds_ratio"] = or_val
                    entry["or_ci_lower"] = or_ci_lower
                    entry["or_ci_upper"] = or_ci_upper
                except Exception:
                    entry["odds_ratio"] = None
                    entry["or_ci_lower"] = None
                    entry["or_ci_upper"] = None
            else:
                entry["model_type"] = "ols"
        except Exception as e:
            entry = {"error_extracting": str(e)}
        summary[key] = entry

    # Aggregate a simple verdict by independent variable across available models
    ivs_of_interest = ["masfem_z", "gender_female", "masfem_mturk_z"]
    aggregation = {}
    for iv in ivs_of_interest:
        rows = [v for k, v in summary.items() if isinstance(v, dict) and v.get("iv") == iv]
        if not rows:
            continue
        agg = {
            "n_models": len(rows),
            "n_positive_coef": sum(1 for r in rows if r.get("coef") is not None and r.get("coef") > 0),
            "n_negative_coef": sum(1 for r in rows if r.get("coef") is not None and r.get("coef") < 0),
            "n_significant_positive": sum(1 for r in rows if r.get("coef") is not None and r.get("coef") > 0 and r.get("significant")),
            "n_significant_negative": sum(1 for r in rows if r.get("coef") is not None and r.get("coef") < 0 and r.get("significant")),
            "models": rows
        }
        # simple rule for judgement:
        if agg["n_significant_positive"] >= 2:
            judgement = "Yes (consistent, statistically significant positive effects across outcomes)"
        elif agg["n_significant_positive"] == 1 and agg["n_models"] >= 2:
            judgement = "Mixed (one significant positive effect, others not significant)"
        elif agg["n_significant_negative"] >= 1:
            judgement = "Mixed/Contrary (some significant negative effects)"
        elif agg["n_significant_positive"] == 0 and agg["n_significant_negative"] == 0:
            judgement = "No evidence (coefficients not statistically significant)"
        else:
            judgement = "Inconclusive"
        agg["judgement"] = judgement
        aggregation[iv] = agg

    # Final overall judgement focusing on primary IV 'masfem_z' then binary indicator
    final_judgement = ""
    if "masfem_z" in aggregation:
        final_judgement = f"masfem_z: {aggregation['masfem_z']['judgement']}"
    elif "gender_female" in aggregation:
        final_judgement = f"gender_female: {aggregation['gender_female']['judgement']}"
    else:
        final_judgement = "No models for masfem_z or gender_female found in model_output."

    # Compose a short human-readable description summarizing the key stats for main IVs
    desc_lines = []
    for iv in ivs_of_interest:
        if iv in aggregation:
            agg = aggregation[iv]
            desc_lines.append(f"{iv}: {agg['n_models']} model(s); "
                              f"{agg['n_positive_coef']} positive coef, {agg['n_negative_coef']} negative coef; "
                              f"{agg['n_significant_positive']} significant positive, {agg['n_significant_negative']} significant negative. Judgement: {agg['judgement']}")
    if not desc_lines:
        description = "No relevant models found for masfem_z or gender_female in provided model_output."
    else:
        description = " | ".join(desc_lines) + f" Overall conclusion (primary IV): {final_judgement}"

    return {
        "object": {
            "model_summaries": summary,
            "summary_by_iv": aggregation,
            "final_judgement": final_judgement
        },
        "description": description
    }