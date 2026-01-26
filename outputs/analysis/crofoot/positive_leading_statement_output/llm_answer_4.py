def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, z-stats, p-values, 95% CIs, and odds ratios (with CIs)
    for the key predictors from a fitted statsmodels logit result (possibly robust/clustered).
    
    Returns:
      {
        "object": dict keyed by term with numeric summary (coef, se, z, p, ci, odds_ratio, or_ci),
        "description": brief explanation of what the numbers mean and how to interpret them
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Try to access standard result attributes in a robust way
    try:
        params = pd.Series(res.params)
    except Exception as e:
        raise ValueError("Could not extract params from model_output: %s" % e)

    # bse/pvalues/conf_int are usually available on the result object (also for robust results)
    # fall back gently if some are missing
    try:
        bse = pd.Series(res.bse)
    except Exception:
        # try extracting sqrt of diagonal of cov_params if available
        try:
            cov = res.cov_params()
            bse = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
        except Exception as e:
            raise ValueError("Could not extract standard errors from model_output: %s" % e)

    try:
        pvalues = pd.Series(res.pvalues)
    except Exception:
        # if pvalues missing, compute from z-statistics if possible
        try:
            zvals = params / bse
            # two-sided p-value from normal approx
            from scipy import stats
            pvalues = pd.Series(2 * (1 - stats.norm.cdf(np.abs(zvals))), index=params.index)
        except Exception as e:
            raise ValueError("Could not extract or compute p-values: %s" % e)

    try:
        conf = res.conf_int(alpha=0.05)
        # conf_int may return ndarray or DataFrame; normalize to DataFrame with columns [lower, upper]
        conf_df = pd.DataFrame(conf, index=params.index)
        conf_df.columns = ['ci_lower', 'ci_upper']
    except Exception:
        # fallback: compute Wald-style CIs from normal approximation
        z_crit = 1.96
        ci_lower = params - z_crit * bse
        ci_upper = params + z_crit * bse
        conf_df = pd.DataFrame({'ci_lower': ci_lower, 'ci_upper': ci_upper})

    # Terms of interest
    terms = ['rel_size_z', 'location_adv_z', 'rel_size_z:location_adv_z']
    # some statsmodels may name interaction 'rel_size_z:location_adv_z' or 'rel_size_z:location_adv_z' (same),
    # include exact match and also try the reverse order with ':' if needed
    available_terms = set(params.index.astype(str))

    # map requested term names to actual available names
    term_map = {}
    for t in terms:
        if t in available_terms:
            term_map[t] = t
        else:
            # try swapping order around ':' (in case statsmodels used other order)
            if ':' in t:
                a, b = t.split(':', 1)
                alt = f'{b}:{a}'
                if alt in available_terms:
                    term_map[t] = alt
                else:
                    # try the '*' style (unlikely here) or with spaces
                    found = None
                    for name in available_terms:
                        if a in name and b in name and ':' in name:
                            found = name
                            break
                    if found:
                        term_map[t] = found
                    else:
                        term_map[t] = None
            else:
                term_map[t] = None

    output = {}
    for logical_name, actual_name in term_map.items():
        if actual_name is None:
            output[logical_name] = {
                'error': f"term '{logical_name}' not found in model. Available terms: {list(params.index)}"
            }
            continue
        coef = float(params.loc[actual_name])
        se = float(bse.loc[actual_name])
        z = coef / se if se != 0 else np.nan
        p = float(pvalues.loc[actual_name])
        ci_low = float(conf_df.loc[actual_name, 'ci_lower'])
        ci_high = float(conf_df.loc[actual_name, 'ci_upper'])
        or_coef = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))

        output[logical_name] = {
            'term_name_in_model': actual_name,
            'coef': coef,
            'se': se,
            'z': z,
            'p_value': p,
            'ci_95_lower': ci_low,
            'ci_95_upper': ci_high,
            'odds_ratio': or_coef,
            'or_ci_95_lower': or_ci_low,
            'or_ci_95_upper': or_ci_high
        }

    # Also provide summaries for control predictors (optional helpful info)
    controls = ['male_diff_z', 'female_diff_z', 'total_n_z']
    controls_out = {}
    for c in controls:
        if c in available_terms:
            coef = float(params.loc[c])
            se = float(bse.loc[c])
            z = coef / se if se != 0 else np.nan
            p = float(pvalues.loc[c])
            ci_low = float(conf_df.loc[c, 'ci_lower'])
            ci_high = float(conf_df.loc[c, 'ci_upper'])
            or_coef = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            controls_out[c] = {
                'term_name_in_model': c,
                'coef': coef,
                'se': se,
                'z': z,
                'p_value': p,
                'ci_95_lower': ci_low,
                'ci_95_upper': ci_high,
                'odds_ratio': or_coef,
                'or_ci_95_lower': or_ci_low,
                'or_ci_95_upper': or_ci_high
            }
        else:
            controls_out[c] = {'error': f"term '{c}' not found in model."}

    # Build a brief interpretation guide
    description_lines = [
        "Extracted estimates for the effect of relative group size (rel_size_z), location advantage (location_adv_z),",
        "and their interaction (rel_size_z:location_adv_z) on the log-odds that the focal group wins.",
        "- For each term you get: coefficient on log-odds scale, robust SE, z-statistic, two-sided p-value, 95% CI,",
        "  and the corresponding odds ratio with 95% CI (exp(coeff)).",
        "- Interpretation guidance:",
        "  * A positive coef means that higher values of that predictor increase the probability that the focal group wins.",
        "  * If the p-value for rel_size_z < 0.05, relative group size has a statistically detectable main effect.",
        "  * If the p-value for location_adv_z < 0.05, being closer to home (home-field advantage) has a detectable effect.",
        "  * If the interaction term is statistically significant (p < 0.05), the effect of relative size depends on location —",
        "    in that case do not interpret rel_size_z or location_adv_z in isolation; compute simple slopes or predicted probabilities",
        "    at representative values of location to understand the conditional effects.",
        "",
        "The 'object' element contains the numeric summaries (see keys). Use those to decide 'yes/no' for each hypothesis",
        "based on p-values and confidence intervals (e.g., p < 0.05 and CI not including 0 indicates a significant effect)."
    ]
    description = "\n".join(description_lines)

    return {'object': {'key_terms': output, 'controls': controls_out}, 'description': description}