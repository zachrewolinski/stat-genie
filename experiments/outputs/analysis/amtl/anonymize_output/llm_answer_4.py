def extract_final_answer(model_output):
    """
    Extract statistics relevant to comparing Homo sapiens vs non-human genera from a fitted
    statsmodels GLM result (possibly with cluster-robust covariances).
    Returns a dictionary with keys:
      - "object": dict of extracted numeric results and a short conclusion label
      - "description": brief explanation of what was extracted and how to interpret it

    Behavior:
      - If an "IsHuman" coefficient exists, that is used as the primary test (interpreted as
        log-odds difference for humans vs non-humans).
      - Otherwise, looks for a parameter name containing "Homo" (case-insensitive).
      - If neither is present, returns all "Genus_" parameters (differences vs the dropped/reference genus).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Attempt to get parameter table pieces robustly
    try:
        params = pd.Series(res.params)  # param name -> estimate
    except Exception as e:
        raise ValueError("Model output does not expose .params: %s" % str(e))

    # p-values, standard errors, conf ints
    pvalues = None
    bse = None
    conf = None
    try:
        pvalues = pd.Series(res.pvalues)
    except Exception:
        # leave as None
        pvalues = None
    try:
        bse = pd.Series(res.bse)
    except Exception:
        bse = None
    try:
        conf = res.conf_int()
        # conf_int may return a numpy array or DataFrame; normalize to DataFrame
        conf = pd.DataFrame(conf, index=params.index, columns=['ci_lower', 'ci_upper'])
    except Exception:
        conf = None

    def make_result_for_param(param_name):
        if param_name not in params.index:
            return None
        coef = float(params.loc[param_name])
        se = float(bse.loc[param_name]) if (bse is not None and param_name in bse.index) else None
        pval = float(pvalues.loc[param_name]) if (pvalues is not None and param_name in pvalues.index) else None
        ci_lower = float(conf.loc[param_name, 'ci_lower']) if (conf is not None and param_name in conf.index) else None
        ci_upper = float(conf.loc[param_name, 'ci_upper']) if (conf is not None and param_name in conf.index) else None
        # For binomial-logit GLM, exponentiate coef to get odds ratio
        try:
            or_val = float(np.exp(coef))
            or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            or_val = None
            or_ci_lower = None
            or_ci_upper = None

        # Simple significance-based conclusion using alpha=0.05 when p-value is available
        if pval is not None:
            if pval < 0.05:
                if coef > 0:
                    conclusion = "Significant: higher AMTL in group indicated by %s (p=%.4g)" % (param_name, pval)
                elif coef < 0:
                    conclusion = "Significant: lower AMTL in group indicated by %s (p=%.4g)" % (param_name, pval)
                else:
                    conclusion = "No difference (coef ~= 0) for %s (p=%.4g)" % (param_name, pval)
            else:
                conclusion = "No statistically significant difference for %s (p=%.4g)" % (param_name, pval)
        else:
            # fallback: use CI if available
            if ci_lower is not None and ci_upper is not None:
                if ci_lower > 0:
                    conclusion = "95%% CI excludes 0 -> higher AMTL for %s" % param_name
                elif ci_upper < 0:
                    conclusion = "95%% CI excludes 0 -> lower AMTL for %s" % param_name
                else:
                    conclusion = "95%% CI includes 0 -> no clear evidence of difference for %s" % param_name
            else:
                conclusion = "Insufficient information to assess significance for %s" % param_name

        return {
            'param_name': param_name,
            'coef': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'odds_ratio': or_val,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'conclusion': conclusion
        }

    # Priority 1: "IsHuman"
    if 'IsHuman' in params.index:
        result = make_result_for_param('IsHuman')
        desc = ("Extracted the coefficient for 'IsHuman' from the fitted GLM. "
                "This coefficient is on the log-odds scale (logit). Exponentiating it "
                "gives the odds ratio for AMTL in humans vs non-humans, controlling for covariates.")
        return {'object': result, 'description': desc}

    # Priority 2: any parameter containing 'Homo' (case-insensitive)
    homo_params = [n for n in params.index if 'homo' in n.lower()]
    if len(homo_params) > 0:
        # If multiple, return all of them
        results = [make_result_for_param(n) for n in homo_params]
        desc = ("Found parameter(s) with 'Homo' in their name. These represent model contrasts "
                "involving Homo (typically vs the reference genus). Each entry contains coefficient, "
                "p-value, 95% CI, odds ratio and a short conclusion.")
        return {'object': results, 'description': desc}

    # Priority 3: any genus dummies (prefix 'Genus_')
    genus_params = [n for n in params.index if n.startswith('Genus_')]
    if len(genus_params) > 0:
        results = [make_result_for_param(n) for n in genus_params]
        desc = ("No explicit 'IsHuman' or 'Homo' parameter found. Returning all genus dummy parameters "
                "(these are contrasts vs the dropped/reference genus). Interpret each coefficient as the "
                "log-odds difference in AMTL for that genus relative to the reference.")
        return {'object': results, 'description': desc}

    # Fallback: return full parameter table summary
    # Provide best-effort summary for all params
    all_params = []
    for n in params.index:
        all_params.append(make_result_for_param(n))
    desc = ("No IsHuman/Homo/Genus_ parameters found. Returning summary for all estimated parameters. "
            "Inspect entries to find the genus- or human-related effect of interest.")
    return {'object': all_params, 'description': desc}