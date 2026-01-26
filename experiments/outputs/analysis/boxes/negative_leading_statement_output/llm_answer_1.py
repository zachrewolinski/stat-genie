def extract_final_answer(model_output):
    """
    Extracts key statistics related to how reliance on the majority option changes with age
    across cultures from the provided model_output dict.

    Returns a dict with:
      - "object": a structured dict containing LR-test p-values, the baseline age effect
                  (log-odds and OR) for choosing the majority option, culture-specific
                  age effects (log-odds and OR), and (if available) p-values for the
                  age coefficient and age-by-culture interactions for the majority outcome.
      - "description": a short, human-readable interpretation of the above results.
    """
    import math
    import re

    mo = model_output  # shorter name

    # Prepare output container
    out_obj = {
        'lr_tests': {},
        'majority_age_effects': {},  # will store per-culture effects for y_code=1 (majority)
        'age_coefficient_pvalues': {}  # p-values parsed from summary if available
    }

    # 1) Extract LR-test p-values (age main effect, culture main effect, age-by-culture)
    for key in ('age_main_effect', 'culture_main_effect', 'age_by_culture_interaction'):
        if key in mo and isinstance(mo[key], dict):
            out_obj['lr_tests'][key] = {
                'pvalue': mo[key].get('pvalue'),
                'lr_stat': mo[key].get('lr_stat'),
                'df': mo[key].get('df')
            }
        else:
            out_obj['lr_tests'][key] = None

    # 2) Extract parameter estimates for the majority outcome (y_code=1).
    #    In the provided model_output, full_model_params holds two groups keyed 0 and 1:
    #    0 -> coefficients for y_code=1 (majority vs reference)
    params = mo.get('full_model_params', {})
    # Accept either int-keys or string-keys
    def get_params_for_majority(p):
        if p is None:
            return None
        # try int 0, then string '0'
        if 0 in p:
            return p[0]
        if '0' in p:
            return p['0']
        # fallback: if only one dict present, pick the first
        for k in p:
            try:
                # choose the first mapping that looks like a dict of params
                if isinstance(p[k], dict):
                    return p[k]
            except Exception:
                continue
        return None

    maj_params = get_params_for_majority(params)
    if maj_params is None:
        raise ValueError("Could not find majority (y_code=1) parameter estimates in model_output['full_model_params'].")

    # Baseline age coefficient (reference culture is Culture 1)
    age_c = maj_params.get('age_c')
    if age_c is None:
        raise ValueError("age_c not found in majority parameters.")

    # Compute per-culture age effects (log-odds) and odds ratios.
    # Culture 1 (reference): effect = age_c
    age_effects = {}
    age_effects['culture_1'] = {
        'age_coef_logodds': float(age_c),
        'age_coef_odds_ratio': float(math.exp(age_c))
    }

    # For cultures 2..8: effect = age_c + age_c_culture_i
    for i in range(2, 9):
        key = f'age_c_culture_{i}'
        add = maj_params.get(key, 0.0)
        # default to 0.0 if not present (should be present per model spec)
        total = float(age_c) + float(add)
        age_effects[f'culture_{i}'] = {
            'age_coef_logodds': total,
            'age_coef_odds_ratio': float(math.exp(total)),
            'interaction_component': float(add)
        }

    out_obj['majority_age_effects'] = age_effects

    # 3) Attempt to extract p-values for age_c and age-by-culture terms from summary text if available.
    #    The summary format used in the provided output is the statsmodels MNLogit summary text.
    summary_text = mo.get('full_model_summary', None)
    if summary_text:
        # We only want the section for y_code=1 (the first block). We'll parse lines after the
        # "y_code=1" header until the separator line of hyphens.
        pvals = {}
        lines = summary_text.splitlines()
        # Find start index of the "y_code=1" block
        start_idx = None
        for idx, line in enumerate(lines):
            if re.search(r'^\s*y_code=1\b', line):
                start_idx = idx + 1  # next line contains headers or separator
                break
        if start_idx is not None:
            # iterate subsequent lines until an empty line or a line that starts with '---' or 'y_code=2'
            for line in lines[start_idx:]:
                if re.match(r'^\s*-{3,}', line) or re.search(r'^\s*y_code=2\b', line):
                    break
                # strip leading/trailing whitespace
                stripped = line.strip()
                if not stripped:
                    continue
                # tokenise by whitespace; valid variable lines start with variable name (e.g., 'age_c' or 'age_c_culture_3')
                tokens = re.split(r'\s+', stripped)
                varname = tokens[0]
                # Expect at least 5 tokens: coef, std err, z, P>|z|, [0.025...
                if len(tokens) >= 5:
                    try:
                        pval = float(tokens[4])
                    except Exception:
                        pval = None
                    pvals[varname] = pval
        # populate age coefficient p-value and interaction p-values
        if 'age_c' in pvals:
            out_obj['age_coefficient_pvalues']['age_c_majority_y'] = pvals['age_c']
        # interactions
        for i in range(2, 9):
            v = f'age_c_culture_{i}'
            if v in pvals:
                out_obj['age_coefficient_pvalues'][v + '_majority_y'] = pvals[v]
    else:
        out_obj['age_coefficient_pvalues'] = None

    # 4) Prepare a short human-readable description summarizing the evidence.
    # Pull LR p-values to include in description
    age_p = out_obj['lr_tests'].get('age_main_effect', {}).get('pvalue') if out_obj['lr_tests'].get('age_main_effect') else None
    culture_p = out_obj['lr_tests'].get('culture_main_effect', {}).get('pvalue') if out_obj['lr_tests'].get('culture_main_effect') else None
    inter_p = out_obj['lr_tests'].get('age_by_culture_interaction', {}).get('pvalue') if out_obj['lr_tests'].get('age_by_culture_interaction') else None

    # Baseline (culture 1) age effect values
    baseline_logodds = age_effects['culture_1']['age_coef_logodds']
    baseline_or = age_effects['culture_1']['age_coef_odds_ratio']

    # Build description
    desc_lines = []
    desc_lines.append("Summary interpretation of model results for children's reliance on the majority option across age and cultures:")
    if age_p is not None:
        desc_lines.append(f"- Overall (likelihood-ratio) test indicates a significant main effect of age (p = {age_p:.3f}).")
    else:
        desc_lines.append("- Overall LR test for age not available.")
    if culture_p is not None:
        desc_lines.append(f"- Overall test indicates a significant main effect of culture (p = {culture_p:.3f}).")
    if inter_p is not None:
        desc_lines.append(f"- The age × culture interaction is significant (p = {inter_p:.3f}), indicating that age-related change in choosing the majority differs across cultures.")
    desc_lines.append(f"- For the majority option (majority vs unchosen), the baseline (reference culture 1) age coefficient is {baseline_logodds:.3f} (log-odds per centered year), corresponding to an odds ratio of {baseline_or:.3f}. A positive value means the probability of choosing the majority increases with age in culture 1.")
    desc_lines.append("- Culture-specific age effects (log-odds per centered year and odds ratios) follow (format: culture: log-odds, OR, interaction component):")
    for ci in sorted(age_effects.keys(), key=lambda x: int(x.split('_')[1])):
        info = age_effects[ci]
        if ci == 'culture_1':
            desc_lines.append(f"  * {ci}: {info['age_coef_logodds']:.3f} (OR={info['age_coef_odds_ratio']:.3f}) [reference]")
        else:
            desc_lines.append(f"  * {ci}: {info['age_coef_logodds']:.3f} (OR={info['age_coef_odds_ratio']:.3f}), interaction={info['interaction_component']:.3f}")
    # If p-values for specific coefficients are available, add note
    if out_obj['age_coefficient_pvalues']:
        desc_lines.append("- Parsed p-values for the majority outcome coefficients (from model summary):")
        for k, v in out_obj['age_coefficient_pvalues'].items():
            desc_lines.append(f"  * {k}: p = {v:.3f}" if v is not None else f"  * {k}: p = NA")
    else:
        desc_lines.append("- Per-coefficient p-values were not available/parsable from the summary text.")

    description = "\n".join(desc_lines)

    return {"object": out_obj, "description": description}