def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM (logistic) result (or its
    robustcov wrapper) for the predictors:
      - RelSize_z
      - DistAdv_z
      - interaction RelSize_z:DistAdv_z

    Returns:
      {
        "object": {
           "coef_table": { var: {coef, se, p, conf_low, conf_high, OR, OR_CI_low, OR_CI_high} , ... },
           "predicted_probabilities": { (rel, dist): prob, ... }  # rel and dist are values in SD units
        },
        "description": "<human-readable summary interpreting results>"
      }
    """
    import numpy as np
    import math

    res = model_output

    # Helper to safely get a result attribute
    def safe_get(attr, default=None):
        return getattr(res, attr) if hasattr(res, attr) else default

    # Try to obtain basic arrays
    try:
        params = safe_get('params', None)
        bse = safe_get('bse', None)
        pvalues = safe_get('pvalues', None)
        conf = safe_get('conf_int', None)
        if callable(conf):
            conf = conf()
    except Exception as e:
        raise ValueError(f"Unable to extract parameters from model_output: {e}")

    if params is None:
        raise ValueError("model_output does not expose .params. Provide a statsmodels results object.")

    # Convert to pandas-like dict for easier lookups (works with pandas Series or numpy arrays with index)
    try:
        # params may be a pandas Series with index
        keys = list(params.index)
    except Exception:
        # fallback: try to use attribute names
        keys = list(params.keys()) if hasattr(params, 'keys') else None

    # Names we expect for the predictors
    possible_inter_names = ['RelSize_z:DistAdv_z', 'DistAdv_z:RelSize_z']  # interaction naming variants
    main_names = ['RelSize_z', 'DistAdv_z']

    # Determine which interaction name exists in the model
    interaction_name = None
    for n in possible_inter_names:
        if n in params.index:
            interaction_name = n
            break
    # If not found, try searching for any name containing both substrings
    if interaction_name is None:
        for n in params.index:
            if 'RelSize_z' in n and 'DistAdv_z' in n and ':' in n:
                interaction_name = n
                break

    # Verify main terms exist
    for mn in main_names:
        if mn not in params.index:
            # attempt to find close matches (in case of different naming)
            matches = [n for n in params.index if mn in n]
            if matches:
                # pick first
                main_names[main_names.index(mn)] = matches[0]
            else:
                raise ValueError(f"Could not find expected predictor '{mn}' in model parameters. Available names: {list(params.index)}")

    # If interaction not present, set to None
    if interaction_name is None:
        # try adding exact string used by patsy sometimes 'RelSize_z:DistAdv_z' already tried; else no interaction
        interaction_name = None

    # Prepare table for variables of interest (including intercept for predictions)
    vars_of_interest = ['Intercept'] + main_names[:]
    if interaction_name is not None:
        vars_of_interest.append(interaction_name)
    # Map intercept name (could be 'Intercept' or 'const')
    intercept_name = None
    for cand in ['Intercept', 'const']:
        if cand in params.index:
            intercept_name = cand
            break
    if intercept_name is None:
        # try to find a parameter name that looks like an intercept
        for n in params.index:
            if n.lower() in ('intercept', 'const'):
                intercept_name = n
                break
    if intercept_name is None:
        # final fallback: use the first parameter as intercept (will still attempt predictions but warn)
        intercept_name = params.index[0]

    # Build coefficient table
    coef_table = {}
    for var in params.index:
        # compute conf interval
        try:
            ci = res.conf_int().loc[var].values if hasattr(res, 'conf_int') else None
        except Exception:
            # conf_int might return numpy array without labels
            try:
                ci_all = res.conf_int()
                # attempt to index by position
                idx = list(params.index).index(var)
                ci = ci_all[idx]
            except Exception:
                ci = [np.nan, np.nan]
        coef = float(params[var])
        se = float(bse[var]) if bse is not None and var in bse.index else float('nan')
        p = float(pvalues[var]) if pvalues is not None and var in pvalues.index else float('nan')
        if ci is None:
            ci_low, ci_high = (float('nan'), float('nan'))
        else:
            try:
                ci_low, ci_high = float(ci[0]), float(ci[1])
            except Exception:
                ci_low, ci_high = (float(ci[0]), float(ci[1]))
        # odds ratio and CI
        OR = math.exp(coef)
        OR_CI_low = math.exp(ci_low) if not math.isnan(ci_low) else float('nan')
        OR_CI_high = math.exp(ci_high) if not math.isnan(ci_high) else float('nan')

        coef_table[var] = {
            'coef_logodds': coef,
            'se': se,
            'p_value': p,
            'conf_low': ci_low,
            'conf_high': ci_high,
            'OR': OR,
            'OR_CI_low': OR_CI_low,
            'OR_CI_high': OR_CI_high
        }

    # Predictions for combinations of RelSize_z and DistAdv_z at values [-1, 0, +1] (SD units)
    # Use MaleDiff_z = FemaleDiff_z = 0 (mean), other predictors not included
    # Build function to compute linear predictor
    def get_param(name):
        if name in params.index:
            return float(params[name])
        return 0.0

    intercept = get_param(intercept_name)
    rel_coef = get_param(main_names[0])
    dist_coef = get_param(main_names[1])
    int_coef = get_param(interaction_name) if interaction_name is not None else 0.0

    def logistic(x):
        return 1.0 / (1.0 + math.exp(-x))

    pred_probs = {}
    for rel in [-1.0, 0.0, 1.0]:
        for dist in [-1.0, 0.0, 1.0]:
            lp = intercept + rel_coef * rel + dist_coef * dist + int_coef * rel * dist
            prob = logistic(lp)
            pred_probs[(rel, dist)] = prob

    # Build human-readable description based on p-values
    alpha = 0.05
    def sig_text(p):
        if math.isnan(p):
            return "p = NA"
        return f"p = {p:.3f}" + (" (significant at 0.05)" if p < alpha else " (not significant at 0.05)")

    # Interpret main effects
    rel_info = coef_table[main_names[0]]
    dist_info = coef_table[main_names[1]]
    interaction_info = coef_table[interaction_name] if interaction_name is not None and interaction_name in coef_table else None

    # Compose interpretation
    lines = []
    # Main effects summary
    lines.append(f"RelSize_z: coef (log-odds) = {rel_info['coef_logodds']:.3f}, se = {rel_info['se']:.3f}, {sig_text(rel_info['p_value'])}; OR = {rel_info['OR']:.3f}, 95%CI = [{rel_info['OR_CI_low']:.3f}, {rel_info['OR_CI_high']:.3f}].")
    lines.append(f"DistAdv_z: coef (log-odds) = {dist_info['coef_logodds']:.3f}, se = {dist_info['se']:.3f}, {sig_text(dist_info['p_value'])}; OR = {dist_info['OR']:.3f}, 95%CI = [{dist_info['OR_CI_low']:.3f}, {dist_info['OR_CI_high']:.3f}].")
    if interaction_info is not None:
        lines.append(f"Interaction ({interaction_name}): coef (log-odds) = {interaction_info['coef_logodds']:.3f}, se = {interaction_info['se']:.3f}, {sig_text(interaction_info['p_value'])}; OR = {interaction_info['OR']:.3f}, 95%CI = [{interaction_info['OR_CI_low']:.3f}, {interaction_info['OR_CI_high']:.3f}].")
    else:
        lines.append("No interaction term found in the fitted model.")

    # Describe predicted probability changes for a simple contrast (RelSize -1 vs +1) at DistAdv = 0
    prob_rel_minus1 = pred_probs[(-1.0, 0.0)]
    prob_rel_plus1 = pred_probs[(1.0, 0.0)]
    delta_rel = prob_rel_plus1 - prob_rel_minus1
    lines.append(f"Predicted probability of focal-group win (controls at mean):\n - RelSize_z = -1, DistAdv_z = 0 -> prob = {prob_rel_minus1:.3f}\n - RelSize_z = +1, DistAdv_z = 0 -> prob = {prob_rel_plus1:.3f}\n => Change ~ {delta_rel:.3f} (difference between -1 and +1 SD in RelSize_z when DistAdv_z = 0).")

    # If interaction present and significant, show how RelSize effect changes with DistAdv
    if interaction_info is not None and not math.isnan(interaction_info['p_value']) and interaction_info['p_value'] < alpha:
        # compute effect of RelSize (slope) at DistAdv = -1,0,1
        slopes = {}
        for dist in [-1.0, 0.0, 1.0]:
            slope = rel_coef + int_coef * dist
            slopes[dist] = slope
        slope_lines = ", ".join([f"at DistAdv {d}: slope = {s:.3f}" for d, s in slopes.items()])
        lines.append("Interaction is significant: the effect (log-odds slope) of RelSize_z depends on DistAdv_z. Slopes of RelSize_z " + slope_lines + ".")
    elif interaction_info is not None:
        lines.append("Interaction term not statistically significant at alpha = 0.05; interpret main effects directly (no evidence that effect of RelSize_z depends on DistAdv_z).")

    description = " ".join(lines)

    # Prepare object to return
    out = {
        "coef_table": coef_table,
        "predicted_probabilities": pred_probs,
        "intercept_name": intercept_name,
        "main_names": {"RelSize": main_names[0], "DistAdv": main_names[1]},
        "interaction_name": interaction_name
    }

    return {"object": out, "description": description}