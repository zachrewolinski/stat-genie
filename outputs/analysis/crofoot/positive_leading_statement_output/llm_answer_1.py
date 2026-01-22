def extract_final_answer(model_output):
    """
    Extract relevant statistics for the effects of:
      - z_size_diff (relative group size advantage)
      - z_rel_distance (contest location advantage)
      - their interaction (z_size_diff : z_rel_distance)
    from the supplied model_output dict.

    Returns:
      {
        "object": {
           "term_name": {
              "coef": float,
              "se": float or None,
              "pvalue": float,
              "ci95": [low, high],
              "odds_ratio": float,
              "odds_ratio_ci95": [low, high]
           }, ...
        },
        "description": str  # brief interpretation of the three terms
      }
    """
    import numpy as np
    import pandas as pd

    # Helper to pick the primary clustered results if available, else primary non-clustered
    res = None
    # Prefer keys with both 'primary' and 'cluster' (robust clustered inference)
    for k in model_output:
        if ('primary' in k.lower()) and ('cluster' in k.lower()):
            res = model_output[k]
            break
    # If none, pick primary (any key containing 'primary')
    if res is None:
        for k in model_output:
            if 'primary' in k.lower():
                res = model_output[k]
                break
    # Last resort: try an item that looks like a fitted results object
    if res is None:
        # take first value
        try:
            res = next(iter(model_output.values()))
        except StopIteration:
            raise ValueError("model_output appears empty")

    # Retrieve parameter names (robust to pandas Series or numpy arrays)
    try:
        param_index = list(res.params.index)
    except Exception:
        # fallback: try to get names from model.exog_names
        try:
            param_index = list(res.model.exog_names)
        except Exception:
            # give up and use positional indices
            param_index = None

    def find_param_name(target):
        """
        Find exact parameter name in the model results that corresponds to the target term.
        target can be a single name (like 'z_size_diff') or a tuple/list for interaction terms.
        For interaction, we'll find any param name containing both substrings.
        """
        if param_index is None:
            return None
        # Interaction: list/tuple of substrings
        if isinstance(target, (list, tuple)):
            subs = target
            for name in param_index:
                if all(sub in name for sub in subs):
                    return name
            # Sometimes interaction uses ':' or '*' or 'x' separators; above covers that via substring matching
            return None
        else:
            # exact or substring match
            # prefer exact match, else substring match
            if target in param_index:
                return target
            for name in param_index:
                if name == target:
                    return name
            for name in param_index:
                if target in name:
                    return name
            return None

    # Terms we care about
    term_specs = {
        'z_size_diff': 'z_size_diff',
        'z_rel_distance': 'z_rel_distance',
        'interaction': ['z_size_diff', 'z_rel_distance']
    }

    results = {}
    missing_terms = []
    for key, spec in term_specs.items():
        pname = find_param_name(spec)
        if pname is None:
            missing_terms.append(key)
            results[key] = {
                "param_name_found": None,
                "coef": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "ci95": [np.nan, np.nan],
                "odds_ratio": np.nan,
                "odds_ratio_ci95": [np.nan, np.nan]
            }
            continue
        # Extract coef, se, pvalue, ci
        try:
            coef = float(res.params[pname])
        except Exception:
            # positional fallback
            try:
                idx = param_index.index(pname)
                coef = float(res.params[idx])
            except Exception:
                coef = np.nan
        # standard error
        se = None
        try:
            se = float(res.bse[pname])
        except Exception:
            try:
                idx = param_index.index(pname)
                se = float(res.bse[idx])
            except Exception:
                se = np.nan
        # p-value
        try:
            pval = float(res.pvalues[pname])
        except Exception:
            try:
                idx = param_index.index(pname)
                pval = float(res.pvalues[idx])
            except Exception:
                pval = np.nan
        # confidence interval
        try:
            ci_obj = res.conf_int()
            # conf_int may be a DataFrame or numpy array
            if hasattr(ci_obj, 'loc'):
                ci_low, ci_high = float(ci_obj.loc[pname, 0]), float(ci_obj.loc[pname, 1])
            else:
                # numpy array fallback
                idx = param_index.index(pname)
                ci_low, ci_high = float(ci_obj[idx, 0]), float(ci_obj[idx, 1])
        except Exception:
            ci_low, ci_high = (np.nan, np.nan)

        # odds ratio and CI (for logistic regression)
        try:
            or_val = float(np.exp(coef))
            or_low, or_high = float(np.exp(ci_low)), float(np.exp(ci_high))
        except Exception:
            or_val, or_low, or_high = (np.nan, np.nan, np.nan)

        results[key] = {
            "param_name_found": pname,
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci95": [ci_low, ci_high],
            "odds_ratio": or_val,
            "odds_ratio_ci95": [or_low, or_high]
        }

    # Create a concise description / interpretation for the three terms
    def interpret_term(r, readable_name):
        if r["param_name_found"] is None:
            return f"{readable_name}: parameter not found in model."
        sig = "statistically significant" if (not np.isnan(r["pvalue"]) and r["pvalue"] < 0.05) else "not statistically significant"
        # direction
        if not np.isnan(r["coef"]):
            if r["coef"] > 0:
                dir_text = "positive association (higher -> higher win probability)"
            elif r["coef"] < 0:
                dir_text = "negative association (higher -> lower win probability)"
            else:
                dir_text = "no association (coef ≈ 0)"
        else:
            dir_text = "effect unknown"
        or_text = f"OR={r['odds_ratio']:.2f}" if not np.isnan(r['odds_ratio']) else "OR=NA"
        ptext = f"p={r['pvalue']:.3f}" if not np.isnan(r['pvalue']) else "p=NA"
        ci_text = ""
        if not any(np.isnan(x) for x in r["ci95"]):
            ci_text = f"95% CI for coef [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] (OR CI [{r['odds_ratio_ci95'][0]:.2f}, {r['odds_ratio_ci95'][1]:.2f}])"
        return f"{readable_name}: {dir_text}; {sig} ({ptext}); {or_text}. {ci_text}"

    desc_lines = []
    desc_lines.append(interpret_term(results['z_size_diff'], "Relative group size advantage (z_size_diff)"))
    desc_lines.append(interpret_term(results['z_rel_distance'], "Location advantage (z_rel_distance)"))
    desc_lines.append(interpret_term(results['interaction'], "Interaction (z_size_diff × z_rel_distance)"))

    description = " ; ".join(desc_lines)

    return {"object": results, "description": description}