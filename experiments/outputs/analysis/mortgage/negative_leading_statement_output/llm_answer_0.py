def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of gender (female) on mortgage acceptance
    from the model_output produced by the modeling function.

    Returns a dictionary:
      - "object": dict with numeric results (odds ratio, p-value, 95% CI, average marginal effect and its p-value/CI when available)
      - "description": brief plain-language interpretation of the effect of being female on mortgage approval

    The function is robust to:
      - model_output being the dict returned by the provided model() function (with keys like
        'female_odds_ratio', 'female_pvalue', 'female_ci_lower', 'female_ci_upper', 'marginal_effects_summary', etc.)
      - or model_output being a dict containing 'model_object' that exposes .params, .pvalues, and .conf_int()
      - or marginal effects present only as a summary text (it will try to parse the female row)
    """
    import math
    import re
    import numpy as np

    # Prepare output structure with defaults
    extracted = {
        'female_odds_ratio': None,
        'female_pvalue': None,
        'female_ci_lower': None,
        'female_ci_upper': None,
        'female_marginal_effect': None,   # in probability points (e.g., 0.0339 = +3.39 percentage points)
        'female_me_pvalue': None,
        'female_me_ci_lower': None,
        'female_me_ci_upper': None,
        'controls': None
    }

    # Helper to safe-convert to float
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if not isinstance(model_output, dict):
        raise ValueError("model_output expected to be a dict as produced by the modeling function")

    # 1) Prefer direct keys if present
    for k_map in [
        ('female_odds_ratio', 'female_odds_ratio'),
        ('female_pvalue', 'female_pvalue'),
        ('female_ci_lower', 'female_ci_lower'),
        ('female_ci_upper', 'female_ci_upper'),
    ]:
        key = k_map[0]
        if key in model_output:
            extracted[key] = _safe_float(model_output[key])

    # 2) If marginal effects summary text present, try to parse the female row
    me_text = model_output.get('marginal_effects_summary') or model_output.get('margeff_summary') or None
    if isinstance(me_text, str):
        # Look for a line that starts with 'female' (possibly leading spaces)
        for line in me_text.splitlines():
            if re.match(r'^\s*female\b', line):
                # tokenize by whitespace and try to parse numbers
                toks = line.split()
                # Expected token order in the produced summary:
                # [name, dy/dx, std err, z, P>|z|, [0.025, 0.975]]
                if len(toks) >= 6:
                    me = _safe_float(toks[1])
                    me_p = _safe_float(toks[4])
                    me_ci_lower = _safe_float(toks[5]) if len(toks) >= 7 else None
                    me_ci_upper = _safe_float(toks[6]) if len(toks) >= 7 else None
                    extracted['female_marginal_effect'] = me
                    extracted['female_me_pvalue'] = me_p
                    extracted['female_me_ci_lower'] = me_ci_lower
                    extracted['female_me_ci_upper'] = me_ci_upper
                break

    # 3) If top-level model_object exists, try to extract coef/pvalues/confint from it
    model_obj = model_output.get('model_object')
    if model_obj is not None:
        # Try to access params and pvalues and conf_int in a few ways
        try:
            params = getattr(model_obj, 'params', None)
            pvalues = getattr(model_obj, 'pvalues', None)
            conf_df = None
            try:
                conf_df = model_obj.conf_int()
            except Exception:
                # conf_int might be a method on an underlying object
                try:
                    conf_df = model_obj._orig_res.conf_int()
                except Exception:
                    conf_df = None

            if params is not None and 'female' in params:
                # compute odds ratio if not already present
                if extracted['female_odds_ratio'] is None:
                    coef = _safe_float(params['female'])
                    if coef is not None:
                        extracted['female_odds_ratio'] = math.exp(coef)
                # pvalue
                if extracted['female_pvalue'] is None and pvalues is not None and 'female' in pvalues:
                    extracted['female_pvalue'] = _safe_float(pvalues['female'])
            if conf_df is not None and 'female' in conf_df.index:
                # conf_df rows may be labeled 0/1 columns or something similar
                try:
                    lower, upper = conf_df.loc['female'].astype(float)
                except Exception:
                    # conf_df could be a numpy array with matching index; try numeric access
                    try:
                        row = conf_df.loc['female']
                        lower, upper = float(row[0]), float(row[1])
                    except Exception:
                        lower, upper = None, None
                if lower is not None and extracted['female_ci_lower'] is None:
                    extracted['female_ci_lower'] = math.exp(lower) if lower is not None else None
                if upper is not None and extracted['female_ci_upper'] is None:
                    extracted['female_ci_upper'] = math.exp(upper) if upper is not None else None
        except Exception:
            pass

    # 4) If odds ratio still missing but coefficient present directly at top-level, compute
    if extracted['female_odds_ratio'] is None and 'female_coef' in model_output:
        coef = _safe_float(model_output.get('female_coef'))
        if coef is not None:
            extracted['female_odds_ratio'] = math.exp(coef)

    # 5) If p-value missing but model_output has summary_text, try to parse it
    if extracted['female_pvalue'] is None:
        # try 'summary_text' or 'summary' string
        for key in ('summary_text', 'summary', 'model_summary'):
            s = model_output.get(key)
            if isinstance(s, str):
                # look for a line starting with 'female'
                for line in s.splitlines():
                    if re.match(r'^\s*female\b', line):
                        toks = line.split()
                        # sample summary line structure in provided output:
                        # female <coef> <std err> <z> <P>|z| <CI lower> <CI upper>
                        if len(toks) >= 5:
                            p = _safe_float(toks[4])
                            if p is not None:
                                extracted['female_pvalue'] = p
                        # also extract CI if not already extracted
                        if extracted['female_ci_lower'] is None and len(toks) >= 7:
                            ci_lower = _safe_float(toks[-2])
                            ci_upper = _safe_float(toks[-1])
                            if ci_lower is not None:
                                extracted['female_ci_lower'] = math.exp(ci_lower) if ci_lower is not None else None
                            if ci_upper is not None:
                                extracted['female_ci_upper'] = math.exp(ci_upper) if ci_upper is not None else None
                        break

    # 6) If marginal effect not found in me_text but model_output contains a 'marginal_effects_summary' as structured dict, use it
    me_obj = model_output.get('marginal_effects') or model_output.get('margeff')
    if extracted['female_marginal_effect'] is None and isinstance(me_obj, dict):
        # look for 'female' key
        fem_me = me_obj.get('female') or me_obj.get('female_marginal') or None
        if isinstance(fem_me, (list, tuple)):
            extracted['female_marginal_effect'] = _safe_float(fem_me[0])
        elif isinstance(fem_me, (int, float)):
            extracted['female_marginal_effect'] = float(fem_me)

    # 7) Controls: try to capture which controls were used (best-effort)
    if extracted['controls'] is None:
        # try to infer from summary_text by looking for column names line or from model_output keys
        possible_controls = []
        # If model_output has a textual summary, find header lines listing variable names (best-effort)
        st = model_output.get('summary_text') or model_output.get('summary') or ''
        if isinstance(st, str) and 'coef' in st and '\n' in st:
            # parse variable names from the summary table lines (lines that start with varname)
            for line in st.splitlines():
                line = line.strip()
                if line and not line.startswith('coef') and re.match(r'^[a-zA-Z_]', line):
                    var = line.split()[0]
                    if var != 'coef' and var != 'const':
                        possible_controls.append(var)
        # fallback: if model_output contained the original model code keys, list them
        if not possible_controls:
            # typical controls used in the modeling function
            possible_controls = [
                'black', 'housing_expense_ratio_z', 'self_employed', 'married',
                'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
                'PI_ratio_z', 'loan_to_value_z', 'denied_PMI'
            ]
        extracted['controls'] = possible_controls

    # Format numeric values with sensible rounding for description
    def _fmt(v, decimals=3):
        if v is None:
            return 'NA'
        try:
            return str(round(float(v), decimals))
        except Exception:
            return str(v)

    # Build description
    desc_parts = []
    # Odds ratio sentence
    if extracted['female_odds_ratio'] is not None:
        descr_or = f"Female applicants have an estimated odds ratio of approval = {_fmt(extracted['female_odds_ratio'],3)}"
        ci_text = ''
        if extracted['female_ci_lower'] is not None and extracted['female_ci_upper'] is not None:
            ci_text = f" (95% CI: {_fmt(extracted['female_ci_lower'],3)} to {_fmt(extracted['female_ci_upper'],3)})"
        p_text = ''
        if extracted['female_pvalue'] is not None:
            p_text = f", p = {_fmt(extracted['female_pvalue'],3)}"
        desc_parts.append(descr_or + ci_text + p_text + ".")
    else:
        desc_parts.append("Odds-ratio for female vs male could not be extracted.")

    # Marginal effect sentence
    if extracted['female_marginal_effect'] is not None:
        # convert to percentage points
        me = extracted['female_marginal_effect']
        me_pct = me * 100.0
        me_txt = f"On the probability scale, being female is associated with an average marginal effect of {round(me_pct,2)} percentage points"
        me_ci_text = ''
        if extracted['female_me_ci_lower'] is not None and extracted['female_me_ci_upper'] is not None:
            me_ci_text = f" (95% CI: {round(extracted['female_me_ci_lower']*100,2)} to {round(extracted['female_me_ci_upper']*100,2)} percentage points)"
        me_p_text = ''
        if extracted['female_me_pvalue'] is not None:
            me_p_text = f", p = {round(extracted['female_me_pvalue'],3)}"
        desc_parts.append(me_txt + me_ci_text + me_p_text + ".")
    else:
        desc_parts.append("Average marginal effect for female could not be extracted.")

    # Controls mention
    desc_parts.append("Model controls (included covariates): " + ", ".join(extracted['controls']) + ".")

    description = " ".join(desc_parts)

    # Construct the object to return (clean numeric values)
    obj = {
        'female_odds_ratio': extracted['female_odds_ratio'],
        'female_odds_ratio_95ci': (extracted['female_ci_lower'], extracted['female_ci_upper']),
        'female_pvalue': extracted['female_pvalue'],
        'female_marginal_effect_prob': extracted['female_marginal_effect'],
        'female_marginal_effect_pct_points': (None if extracted['female_marginal_effect'] is None else extracted['female_marginal_effect'] * 100.0),
        'female_marginal_effect_95ci': (extracted['female_me_ci_lower'], extracted['female_me_ci_upper']),
        'female_marginal_effect_pvalue': extracted['female_me_pvalue'],
        'controls': extracted['controls']
    }

    return {'object': obj, 'description': description}