def extract_final_answer(model_output):
    """
    Extract key statistics for predictors age, sex, and help_binary from a fitted statsmodels model
    (MixedLMResultsWrapper, RegressionResultsWrapper, or similar).

    Returns:
      {
        "object": {
            "age": {"coef": ..., "se": ..., "p": ..., "ci_lower": ..., "ci_upper": ..., "significant": True/False, "interpretation": "..."},
            "sex_m_vs_f": { ... },   # if sex parameter present (male vs female)
            "help_binary": { ... }
        },
        "description": "Concise textual summary of the effects and statistical significance."
      }
    """
    import numpy as np

    # Helper to safely get attributes or return None
    def _get_attr(obj, name):
        return getattr(obj, name, None)

    # Try to obtain parameter estimates, standard errors, p-values, and conf int
    params = _get_attr(model_output, "params")
    bse = _get_attr(model_output, "bse")
    pvalues = _get_attr(model_output, "pvalues")
    # conf_int might be a method or attribute
    try:
        conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    # If any of these are pandas Series/DataFrame, convert to dict/ndarray for indexing
    try:
        param_names = list(params.index) if params is not None and hasattr(params, "index") else (list(params.keys()) if isinstance(params, dict) else None)
    except Exception:
        param_names = None

    # Convenience function to extract stats for a parameter name (exact match or best match)
    def find_param_key(targets):
        """
        targets: list of possible substrings to match parameter name.
        Returns the first matching parameter name from params index, or None.
        """
        if params is None:
            return None
        names = list(params.index) if hasattr(params, "index") else list(params.keys()) if isinstance(params, dict) else None
        if not names:
            return None
        # Exact match first
        for t in targets:
            if t in names:
                return t
        # Otherwise find by substring presence (longer names first to avoid partial matches)
        for t in targets:
            for n in names:
                if t in n:
                    return n
        # As a last resort, check any name that contains the short token parts
        for t in targets:
            for n in names:
                if any(tok in n for tok in t.split()):
                    return n
        return None

    results = {}

    # 1) Age
    age_key = find_param_key(["age"])
    if age_key is not None:
        coef = float(params[age_key])
        se = float(bse[age_key]) if bse is not None and age_key in bse.index else (float(bse[age_key]) if bse is not None and isinstance(bse, dict) and age_key in bse else None)
        p = float(pvalues[age_key]) if pvalues is not None and age_key in pvalues.index else (float(pvalues[age_key]) if pvalues is not None and isinstance(pvalues, dict) and age_key in pvalues else None)
        if conf_int is not None:
            try:
                # conf_int may be DataFrame-like with index matching params
                if hasattr(conf_int, "loc"):
                    ci_lower = float(conf_int.loc[age_key].iloc[0])
                    ci_upper = float(conf_int.loc[age_key].iloc[1])
                else:
                    # ndarray with same ordering as params
                    ci_idx = param_names.index(age_key) if param_names else None
                    if ci_idx is not None:
                        ci_lower = float(conf_int[ci_idx, 0])
                        ci_upper = float(conf_int[ci_idx, 1])
                    else:
                        ci_lower = ci_upper = None
            except Exception:
                ci_lower = ci_upper = None
        else:
            ci_lower = ci_upper = None

        results["age"] = {
            "param_name": age_key,
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": (p is not None and p < 0.05)
        }
    else:
        results["age"] = {"error": "age parameter not found in model output."}

    # 2) Sex (categorical). Look for parameter naming patterns like 'C(sex)[T.m]' or 'sex[T.m]' or 'sex_m'
    sex_key = find_param_key(["C(sex)", "sex", "C(sex)[T.m]", "C(sex)[T.male]", "sex[T.m]"])
    # We need to ensure we pick the parameter that represents male vs female, not some hammer/other param.
    # Narrow down to names that include 'sex' and either 'T.' or just 'm'/'male'
    chosen_sex_key = None
    if sex_key is not None:
        # collect candidate names containing 'sex'
        names = list(params.index) if hasattr(params, "index") else []
        sex_candidates = [n for n in names if "sex" in n]
        # prefer those with 'T.' (statsmodels coding) or 'male' or 'm'
        prioritized = [n for n in sex_candidates if ("T." in n or "male" in n.lower() or "m" in n.split() or "[T." in n)]
        if prioritized:
            chosen_sex_key = prioritized[0]
        elif sex_candidates:
            chosen_sex_key = sex_candidates[0]
        else:
            chosen_sex_key = None

    if chosen_sex_key is not None:
        coef = float(params[chosen_sex_key])
        se = float(bse[chosen_sex_key]) if bse is not None and chosen_sex_key in bse.index else (float(bse[chosen_sex_key]) if bse is not None and isinstance(bse, dict) and chosen_sex_key in bse else None)
        p = float(pvalues[chosen_sex_key]) if pvalues is not None and chosen_sex_key in pvalues.index else (float(pvalues[chosen_sex_key]) if pvalues is not None and isinstance(pvalues, dict) and chosen_sex_key in pvalues else None)
        if conf_int is not None:
            try:
                if hasattr(conf_int, "loc"):
                    ci_lower = float(conf_int.loc[chosen_sex_key].iloc[0])
                    ci_upper = float(conf_int.loc[chosen_sex_key].iloc[1])
                else:
                    ci_idx = param_names.index(chosen_sex_key) if param_names else None
                    if ci_idx is not None:
                        ci_lower = float(conf_int[ci_idx, 0])
                        ci_upper = float(conf_int[ci_idx, 1])
                    else:
                        ci_lower = ci_upper = None
            except Exception:
                ci_lower = ci_upper = None
        else:
            ci_lower = ci_upper = None

        # Create a readable label: typically this coef is effect of male relative to baseline (female)
        results["sex_m_vs_f"] = {
            "param_name": chosen_sex_key,
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": (p is not None and p < 0.05),
            "interpretation_note": "Positive coef means males have higher log_efficiency than the reference (likely females)."
        }
    else:
        results["sex_m_vs_f"] = {"error": "sex parameter not found in model output."}

    # 3) Help binary
    help_key = find_param_key(["help_binary", "help"])
    if help_key is not None:
        coef = float(params[help_key])
        se = float(bse[help_key]) if bse is not None and help_key in bse.index else (float(bse[help_key]) if bse is not None and isinstance(bse, dict) and help_key in bse else None)
        p = float(pvalues[help_key]) if pvalues is not None and help_key in pvalues.index else (float(pvalues[help_key]) if pvalues is not None and isinstance(pvalues, dict) and help_key in pvalues else None)
        if conf_int is not None:
            try:
                if hasattr(conf_int, "loc"):
                    ci_lower = float(conf_int.loc[help_key].iloc[0])
                    ci_upper = float(conf_int.loc[help_key].iloc[1])
                else:
                    ci_idx = param_names.index(help_key) if param_names else None
                    if ci_idx is not None:
                        ci_lower = float(conf_int[ci_idx, 0])
                        ci_upper = float(conf_int[ci_idx, 1])
                    else:
                        ci_lower = ci_upper = None
            except Exception:
                ci_lower = ci_upper = None
        else:
            ci_lower = ci_upper = None

        results["help_binary"] = {
            "param_name": help_key,
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": (p is not None and p < 0.05),
            "interpretation_note": "Positive coef means receiving help is associated with higher log_efficiency."
        }
    else:
        results["help_binary"] = {"error": "help_binary parameter not found in model output."}

    # Build a concise human-readable description
    desc_parts = []
    # Age description
    if "error" not in results["age"]:
        a = results["age"]
        sign = "increase" if a["coef"] > 0 else ("decrease" if a["coef"] < 0 else "no change")
        sig = "statistically significant (p < 0.05)" if a["significant"] else "not statistically significant (p >= 0.05 or p missing)"
        desc_parts.append(f"Age: coef={a['coef']:.4g}, {sign} in log_efficiency per year; {sig}.")
    else:
        desc_parts.append("Age: parameter not found.")

    # Sex description
    if "error" not in results["sex_m_vs_f"]:
        s = results["sex_m_vs_f"]
        sign = "higher" if s["coef"] > 0 else ("lower" if s["coef"] < 0 else "no difference")
        sig = "statistically significant" if s["significant"] else "not statistically significant"
        desc_parts.append(f"Sex (male vs reference): coef={s['coef']:.4g}, males {sign} in log_efficiency compared to reference; {sig} (p={s['p_value']:.4g}).")
    else:
        desc_parts.append("Sex: parameter not found.")

    # Help description
    if "error" not in results["help_binary"]:
        h = results["help_binary"]
        sign = "higher" if h["coef"] > 0 else ("lower" if h["coef"] < 0 else "no difference")
        sig = "statistically significant" if h["significant"] else "not statistically significant"
        desc_parts.append(f"Help received: coef={h['coef']:.4g}, sessions with help have {sign} log_efficiency compared to sessions without help; {sig} (p={h['p_value']:.4g}).")
    else:
        desc_parts.append("Help: parameter not found.")

    description = " ".join(desc_parts)

    return {"object": results, "description": description}