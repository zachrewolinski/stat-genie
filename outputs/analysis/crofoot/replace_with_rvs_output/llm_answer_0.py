def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels binary-logit results object
    for the model:
      win ~ RelSize_std * Location_Focal + MaleAdv_std + FemaleAdv_std + DistDiff_std + C(dyad)

    Returns a dictionary with:
      - "object": dict of extracted numeric results (coefficients, SEs, p-values,
                  95% CIs, odds ratios) and marginal effects of relative size
                  when Location_Focal==0 and Location_Focal==1.
      - "description": textual explanation of what each extracted value means.

    The function is robust to whether the model_output already has cluster-robust
    covariances (i.e., results from get_robustcov_results) or is a plain fitted result.
    """
    import numpy as np
    import math
    import pandas as pd

    # Helper: normal two-sided p-value from z
    def z_to_p(z):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # Get parameter names and values
    try:
        params = model_output.params.copy()
    except Exception as e:
        raise ValueError(f"Could not read params from model_output: {e}")

    param_names = [str(n) for n in params.index]

    # Get covariance matrix (robust if available)
    try:
        cov = model_output.cov_params()
    except Exception:
        # try attribute .normalized_cov_params or fallback to outer product of bse
        try:
            bse = model_output.bse
            cov = pd.DataFrame(np.diag(bse.values**2), index=bse.index, columns=bse.index)
        except Exception as e:
            raise ValueError(f"Could not obtain covariance matrix nor bse: {e}")

    cov = pd.DataFrame(cov)  # ensure DataFrame for .loc operations
    cov.index = cov.index.astype(str)
    cov.columns = cov.columns.astype(str)

    # Flexible parameter name matching
    def find_param(contain_all, exclude_any=None):
        exclude_any = exclude_any or []
        for n in param_names:
            if all(tok in n for tok in contain_all) and not any(tok in n for tok in exclude_any):
                return n
        return None

    # Find main terms and interaction (robust to different naming conventions)
    rel_name = find_param(['RelSize_std'], exclude_any=['Location_Focal', ':'])
    loc_name = find_param(['Location_Focal'], exclude_any=['RelSize_std', ':'])
    # interaction must contain both tokens (could be 'RelSize_std:Location_Focal' or 'RelSize_std:Location_Focal[T.1]' etc.)
    inter_name = None
    for n in param_names:
        if ('RelSize_std' in n) and ('Location_Focal' in n) and n != rel_name and n != loc_name:
            inter_name = n
            break

    # If we couldn't find main names exactly, attempt looser matches
    if rel_name is None:
        # try any name that contains 'RelSize_std' (even if it also includes ':')
        rel_name = find_param(['RelSize_std'])
    if loc_name is None:
        loc_name = find_param(['Location_Focal'])

    # Prepare result container
    results = {}

    def safe_get_coef(name):
        if name is None:
            return None, None
        coef = float(params.loc[name])
        se = None
        try:
            se = math.sqrt(float(cov.loc[name, name]))
        except Exception:
            # try model_output.bse lookup
            try:
                se = float(model_output.bse.loc[name])
            except Exception:
                se = None
        return coef, se

    # Extract coefficients and SEs
    beta_rel, se_rel = safe_get_coef(rel_name)
    beta_loc, se_loc = safe_get_coef(loc_name)
    beta_int, se_int = safe_get_coef(inter_name)

    # For clarity, record the exact param names used
    results['param_names_used'] = {'RelSize': rel_name, 'Location_Focal': loc_name, 'Interaction': inter_name}

    # Compute z, p, CI, OR for a single coefficient
    def summarize_coef(name, beta, se):
        if beta is None:
            return None
        if se is None:
            # fallback: we can still return coef without SE/pval
            return {
                'coef': beta,
                'se': None,
                'z': None,
                'p': None,
                'ci_95': (None, None),
                'odds_ratio': math.exp(beta),
                'or_ci_95': (None, None)
            }
        z = beta / se
        p = z_to_p(z)
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
        or_ = math.exp(beta)
        or_ci = (math.exp(ci_low), math.exp(ci_high))
        return {
            'coef': beta,
            'se': se,
            'z': z,
            'p': p,
            'ci_95': (ci_low, ci_high),
            'odds_ratio': or_,
            'or_ci_95': or_ci
        }

    results['RelSize_std'] = summarize_coef(rel_name, beta_rel, se_rel)
    results['Location_Focal'] = summarize_coef(loc_name, beta_loc, se_loc)
    results['Interaction'] = summarize_coef(inter_name, beta_int, se_int)

    # Compute marginal effect of RelSize when Location_Focal == 0 (baseline) and == 1
    # Baseline (Location_Focal=0): effect = beta_rel
    # When Location_Focal=1: effect = beta_rel + beta_int
    marginal = {}
    if beta_rel is not None:
        marginal['Location_Focal=0'] = {
            'beta': beta_rel,
            'se': se_rel
        }
    else:
        marginal['Location_Focal=0'] = None

    if beta_rel is not None:
        if beta_int is not None:
            # compute combined variance: var_rel + var_int + 2*cov(rel,int)
            try:
                cov_rel_int = float(cov.loc[rel_name, inter_name])
            except Exception:
                cov_rel_int = 0.0
            var_comb = (se_rel**2 if se_rel is not None else 0.0) + (se_int**2 if se_int is not None else 0.0) + 2.0 * cov_rel_int
            se_comb = math.sqrt(var_comb) if var_comb >= 0 else None
            beta_comb = beta_rel + beta_int
            if se_comb is not None:
                z = beta_comb / se_comb
                p = z_to_p(z)
                ci_low = beta_comb - 1.96 * se_comb
                ci_high = beta_comb + 1.96 * se_comb
            else:
                z = p = ci_low = ci_high = None
            marginal['Location_Focal=1'] = {
                'beta': beta_comb,
                'se': se_comb,
                'z': z,
                'p': p,
                'ci_95': (ci_low, ci_high),
                'odds_ratio': math.exp(beta_comb) if beta_comb is not None else None,
                'or_ci_95': (math.exp(ci_low), math.exp(ci_high)) if (ci_low is not None and ci_high is not None) else (None, None)
            }
        else:
            # no interaction term found => same effect for both locations
            marginal['Location_Focal=1'] = {
                'beta': beta_rel,
                'se': se_rel,
                'z': (beta_rel / se_rel) if se_rel is not None else None,
                'p': (z_to_p(beta_rel / se_rel) if se_rel is not None else None),
                'ci_95': ((beta_rel - 1.96 * se_rel, beta_rel + 1.96 * se_rel) if se_rel is not None else (None, None)),
                'odds_ratio': math.exp(beta_rel),
                'or_ci_95': (math.exp(beta_rel - 1.96 * se_rel), math.exp(beta_rel + 1.96 * se_rel)) if se_rel is not None else (None, None)
            }
    else:
        marginal['Location_Focal=1'] = None

    results['marginal_effects_on_log_odds'] = marginal

    # Also include p-values and a compact verdict fields (but do not over-interpret)
    verdict = {}
    # Significant effect of RelSize at baseline?
    if results['RelSize_std'] is not None and results['RelSize_std']['p'] is not None:
        verdict['RelSize_significant_at_baseline'] = (results['RelSize_std']['p'] < 0.05)
    else:
        verdict['RelSize_significant_at_baseline'] = None

    # Significant interaction?
    if results['Interaction'] is not None and results['Interaction']['p'] is not None:
        verdict['Interaction_significant'] = (results['Interaction']['p'] < 0.05)
    else:
        verdict['Interaction_significant'] = None

    # Significant effect of RelSize when Location_Focal==1?
    me1 = results['marginal_effects_on_log_odds'].get('Location_Focal=1')
    if me1 is not None and me1.get('p') is not None:
        verdict['RelSize_significant_at_focal_location'] = (me1['p'] < 0.05)
    else:
        verdict['RelSize_significant_at_focal_location'] = None

    results['verdict'] = verdict

    # Build a human-readable description (concise)
    description_lines = []
    description_lines.append("Extracted model coefficients and inference for key terms from the fitted logit model.")
    description_lines.append(f"- 'RelSize_std' param used: {rel_name}")
    description_lines.append(f"- 'Location_Focal' param used: {loc_name}")
    description_lines.append(f"- interaction param used: {inter_name}")
    description_lines.append("")
    description_lines.append("Interpretation guide:")
    description_lines.append("- 'RelSize_std' coef = effect (log-odds) of a 1 SD increase in focal group's relative size when Location_Focal == 0 (i.e., baseline location).")
    description_lines.append("- If an interaction term is present, the effect of RelSize when Location_Focal == 1 equals (RelSize_std coef + interaction coef).")
    description_lines.append("- Reported p-values are two-sided Wald tests; OR = exp(coef) gives multiplicative change in odds.")
    description_lines.append("")
    description_lines.append("Summary of extracted numeric results is available under the 'object' key as a dict.")
    description = "\n".join(description_lines)

    return {"object": results, "description": description}