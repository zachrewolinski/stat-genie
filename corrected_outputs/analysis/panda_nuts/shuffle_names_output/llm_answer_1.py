def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels MixedLMResults (or wrapper) object
    to answer how age, sex, and receiving help influence nut-cracking efficiency.

    Returns a dict with:
      - "object": a dict mapping terms of interest (age, sex, help, interactions)
                  to extracted statistics (coef, se, pvalue, 95% CI, significant).
      - "description": a short, data-driven interpretation of those statistics
                       in the context of the task.
    """
    import re
    import numpy as np
    from math import isfinite

    # Try to import scipy.stats for p-value calculation fallback
    try:
        from scipy import stats
    except Exception:
        stats = None

    # Helper: get attributes robustly, fall back if missing
    # params, bse, pvalues, conf_int
    try:
        params = model_output.params
    except Exception:
        raise ValueError("Could not extract params from model_output")

    try:
        bse = model_output.bse
    except Exception:
        bse = None

    # p-values: use available attribute, otherwise approximate by normal z
    pvalues = None
    if hasattr(model_output, "pvalues"):
        try:
            pvalues = model_output.pvalues
        except Exception:
            pvalues = None

    # conf_int: try method, otherwise compute approximate 95% CI using 1.96*se
    conf_int = None
    if hasattr(model_output, "conf_int"):
        try:
            conf_int = model_output.conf_int()
        except Exception:
            conf_int = None

    # Convert Series/DataFrame-like to dict-like mapping term -> value
    # Ensure keys are strings
    params_dict = {str(k): float(v) for k, v in params.items()}

    bse_dict = None
    if bse is not None:
        try:
            bse_dict = {str(k): float(v) for k, v in bse.items()}
        except Exception:
            bse_dict = None

    pval_dict = None
    if pvalues is not None:
        try:
            pval_dict = {str(k): float(v) for k, v in pvalues.items()}
        except Exception:
            pval_dict = None

    ci_dict = None
    if conf_int is not None:
        # conf_int may be a DataFrame or ndarray with index matching params
        try:
            # If it's a DataFrame with index
            ci_dict = {}
            for idx, row in conf_int.iterrows():
                ci_dict[str(idx)] = (float(row[0]), float(row[1]))
        except Exception:
            try:
                # If it's an ndarray with same order as params.index
                keys = list(params.index)
                ci_arr = np.asarray(conf_int)
                ci_dict = {str(k): (float(ci_arr[i, 0]), float(ci_arr[i, 1])) for i, k in enumerate(keys)}
            except Exception:
                ci_dict = None

    # If p-values missing but we have params and bse, approximate via normal z
    if pval_dict is None and bse_dict is not None:
        pval_dict = {}
        for k, coef in params_dict.items():
            se = bse_dict.get(k, None)
            if se is None or se == 0 or not isfinite(se):
                pval_dict[k] = None
            else:
                z = coef / se
                if stats is not None:
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    # Approximate using math.erf if scipy not available
                    from math import erf, sqrt
                    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
                pval_dict[k] = float(p)

    # If CI missing but we have params and bse, compute approx 95% CI
    if ci_dict is None and bse_dict is not None:
        ci_dict = {}
        for k, coef in params_dict.items():
            se = bse_dict.get(k, None)
            if se is None or not isfinite(se):
                ci_dict[k] = (None, None)
            else:
                lo = coef - 1.96 * se
                hi = coef + 1.96 * se
                ci_dict[k] = (float(lo), float(hi))

    # Identify terms of interest by name patterns.
    # We search through the parameter names flexibly because statsmodels can use
    # names like 'C(sex)[T.M]', 'C(help_received)[T.1]', 'age_years',
    # 'age_years:C(help_received)[T.1]', 'C(sex)[T.M]:C(help_received)[T.1]' etc.
    terms = list(params_dict.keys())

    def find_terms(contain_all=None, contain_any=None, start_with=None):
        res = []
        for t in terms:
            ok = True
            if contain_all:
                for s in contain_all:
                    if s not in t:
                        ok = False
                        break
            if contain_any and ok:
                ok = any(s in t for s in contain_any)
            if start_with and ok:
                ok = any(t.startswith(s) for s in start_with)
            if ok:
                res.append(t)
        return res

    # Main effect for age
    age_terms = find_terms(contain_all=["age_years"], contain_any=None)
    # Main effect for sex (any C(sex) terms not part of an interaction)
    sex_main_terms = [t for t in find_terms(contain_all=["C(sex)"]) if "help_received" not in t and ":" not in t]
    # Main effect for help_received
    help_main_terms = [t for t in find_terms(contain_all=["C(help_received)"]) if "sex" not in t and "age_years" not in t and ":" not in t]
    # Interactions
    age_help_terms = [t for t in terms if ("age_years" in t and "help_received" in t)]
    sex_help_terms = [t for t in terms if ("C(sex)" in t and "help_received" in t)]

    # Prepare extracted results for relevant terms
    extracted = {}

    def make_entry(term):
        coef = params_dict.get(term, None)
        se = (bse_dict.get(term) if bse_dict is not None else None)
        p = (pval_dict.get(term) if pval_dict is not None else None)
        ci = (ci_dict.get(term) if ci_dict is not None else (None, None))
        sig = None
        if p is not None:
            sig = bool(p < 0.05)
        return {
            "term": term,
            "coef": float(coef) if coef is not None else None,
            "se": float(se) if se is not None else None,
            "p_value": float(p) if p is not None else None,
            "ci_95": (float(ci[0]) if ci is not None and ci[0] is not None else None,
                      float(ci[1]) if ci is not None and ci[1] is not None else None),
            "significant_at_0.05": sig
        }

    # Add identified terms
    for t in age_terms:
        extracted.setdefault("age_terms", []).append(make_entry(t))
    for t in sex_main_terms:
        extracted.setdefault("sex_main_terms", []).append(make_entry(t))
    for t in help_main_terms:
        extracted.setdefault("help_main_terms", []).append(make_entry(t))
    for t in age_help_terms:
        extracted.setdefault("age_help_interactions", []).append(make_entry(t))
    for t in sex_help_terms:
        extracted.setdefault("sex_help_interactions", []).append(make_entry(t))

    # Also include full coefficient table summary for transparency (optional)
    # but keep it compact: map term -> (coef, p)
    coef_table = {}
    for t in terms:
        coef_table[t] = {
            "coef": params_dict.get(t, None),
            "p_value": (pval_dict.get(t, None) if pval_dict is not None else None)
        }
    extracted["coef_table_brief"] = coef_table

    # Build a concise description / interpretation based on p-values and interactions
    lines = []
    alpha = 0.05

    # Age interpretation
    if extracted.get("age_terms"):
        # If multiple age terms (unlikely), summarize first as main effect
        age_entry = extracted["age_terms"][0]
        age_sig = age_entry["significant_at_0.05"]
        if extracted.get("age_help_interactions"):
            # If any age:help interaction significant, note that
            any_age_help_sig = any(e["significant_at_0.05"] for e in extracted["age_help_interactions"] if e["significant_at_0.05"] is not None)
            if any_age_help_sig:
                lines.append("There is evidence that the effect of age on nut-cracking efficiency depends on whether help was received (significant age × help interaction). Interpret the age coefficient cautiously because the age effect differs between help conditions.")
            else:
                if age_sig:
                    lines.append(f"Age is associated with efficiency: coef = {age_entry['coef']:.4g}, p = {age_entry['p_value']:.4g} (95% CI {age_entry['ci_95']}). This indicates a {('positive' if age_entry['coef']>0 else 'negative')} change in nuts/sec per additional year of age, averaged across help conditions.")
                else:
                    lines.append(f"No strong evidence for a main effect of age (coef = {age_entry['coef']:.4g}, p = {age_entry['p_value']:.4g}). However interactions with help were tested and should be checked.")
        else:
            if age_sig:
                lines.append(f"Age is significantly associated with efficiency (coef = {age_entry['coef']:.4g}, p = {age_entry['p_value']:.4g}; 95% CI {age_entry['ci_95']}).")
            else:
                lines.append(f"No statistically significant main effect of age (coef = {age_entry['coef']:.4g}, p = {age_entry['p_value']:.4g}).")

    else:
        lines.append("No explicit 'age_years' main-effect parameter found in the fitted model output.")

    # Sex interpretation
    if extracted.get("sex_main_terms"):
        for e in extracted["sex_main_terms"]:
            label = e["term"]
            # try to parse which level is compared: C(sex)[T.<level>]
            m = re.search(r"C\(sex\)\)\[T\.?([^\]]+)\]", label)
            # fallback: just show label
            level = None
            if "C(sex)" in label:
                # find stuff between [T. and ]
                m2 = re.search(r"C\(sex\)\)\[T\.?([^\]]+)\]", label) or re.search(r"C\(sex\)\[T\.?([^\]]+)\]", label)
                if m2:
                    level = m2.group(1)
            if e["significant_at_0.05"]:
                lines.append(f"Sex effect ({label}) is significant: coef = {e['coef']:.4g}, p = {e['p_value']:.4g} (95% CI {e['ci_95']}). This indicates that the named sex level differs from the reference in efficiency.")
            else:
                lines.append(f"No strong evidence of a main sex effect for parameter '{label}' (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}).")
    else:
        lines.append("No standalone main-effect parameter for sex (C(sex)) was found (it may be represented only in interactions).")

    # Help interpretation
    if extracted.get("help_main_terms"):
        for e in extracted["help_main_terms"]:
            if e["significant_at_0.05"]:
                lines.append(f"Receiving help main effect ({e['term']}) is significant: coef = {e['coef']:.4g}, p = {e['p_value']:.4g}; receiving help is associated with a {('higher' if e['coef']>0 else 'lower')} nuts/sec on average.")
            else:
                lines.append(f"No strong evidence for a main effect of receiving help ({e['term']}) (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}).")
    else:
        lines.append("No standalone main-effect parameter for help_received was found (it may be represented only in interactions).")

    # Interactions summary
    if extracted.get("age_help_interactions"):
        for e in extracted["age_help_interactions"]:
            if e["significant_at_0.05"]:
                lines.append(f"Significant interaction found: {e['term']} (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}) — the slope of age differs between help conditions.")
            else:
                lines.append(f"No evidence for interaction {e['term']} (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}).")
    if extracted.get("sex_help_interactions"):
        for e in extracted["sex_help_interactions"]:
            if e["significant_at_0.05"]:
                lines.append(f"Significant interaction found: {e['term']} (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}) — the sex difference in efficiency depends on help received.")
            else:
                lines.append(f"No evidence for interaction {e['term']} (coef = {e['coef']:.4g}, p = {e['p_value']:.4g}).")

    # Combine into description
    description = " ".join(lines)

    return {"object": extracted, "description": description}