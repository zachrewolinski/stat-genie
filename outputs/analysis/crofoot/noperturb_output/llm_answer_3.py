def extract_final_answer(model_output):
    """
    Extracts key statistics from the fitted (clustered) logistic regression result and returns:
      - "object": a dictionary with coefficients, SEs, p-values, odds ratios and 95% CIs
                  for the main predictors (RelSize_z, DistDiff_z), their interaction, and Location dummies;
                  plus simple-slope estimates for the effect of RelSize_z at DistDiff_z = -1, 0, +1 (SD).
      - "description": a short plain-language interpretation of those results in the context of the task.
    
    The function is robust to either a statsmodels results object with get_robustcov_results OR the
    ClusteredResults-like wrapper returned by the modeling function in the prompt.
    """
    import numpy as np
    import pandas as pd
    from math import exp
    try:
        # prefer scipy for p-values if we need to compute them
        from scipy.stats import norm as _norm
        norm_cdf = _norm.cdf
    except Exception:
        # fallback to an approximation using numpy (less ideal); this path is unlikely
        def _erf(x):
            # approximate normal cdf via error function
            return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))
        norm_cdf = _erf

    # Helper: obtain params, bse, pvalues, cov matrix (if available)
    # model_output may already be a wrapper providing .params, .bse, .pvalues, .cov_params()
    # or a statsmodels object with get_robustcov_results applied.
    # We'll attempt to access attributes defensively.
    # Params
    if hasattr(model_output, 'params'):
        params = model_output.params
    else:
        raise ValueError("model_output has no 'params' attribute.")

    # ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Could not coerce model_output.params into a pandas Series.")

    # Standard errors
    if hasattr(model_output, 'bse'):
        bse = model_output.bse
        if not isinstance(bse, pd.Series):
            try:
                bse = pd.Series(bse, index=params.index)
            except Exception:
                bse = None
    else:
        bse = None

    # p-values
    if hasattr(model_output, 'pvalues'):
        pvalues = model_output.pvalues
        if not isinstance(pvalues, pd.Series):
            try:
                pvalues = pd.Series(pvalues, index=params.index)
            except Exception:
                pvalues = None
    else:
        pvalues = None

    # covariance matrix (for joint variance / delta method)
    cov_df = None
    try:
        cov_raw = model_output.cov_params()
        # cov_raw may be DataFrame or ndarray; convert to DataFrame aligned with params
        if isinstance(cov_raw, pd.DataFrame):
            cov_df = cov_raw.reindex(index=params.index, columns=params.index)
        else:
            cov_df = pd.DataFrame(cov_raw, index=params.index, columns=params.index)
    except Exception:
        cov_df = None

    # If bse or pvalues missing, compute from cov_df if available
    if (bse is None or any(pd.isnull(bse))) and cov_df is not None:
        bse = pd.Series(np.sqrt(np.diag(cov_df.values)), index=params.index)
    if pvalues is None:
        # compute z and two-sided normal p-values
        if bse is None:
            raise ValueError("Cannot compute p-values: both pvalues and bse/covariance are unavailable.")
        z = params.values / bse.values
        pvals = 2 * (1 - norm_cdf(np.abs(z)))
        pvalues = pd.Series(pvals, index=params.index)

    # prepare results container
    results = {}

    def collect_term(name):
        """Return dict with coef, se, z, p, OR, 95% CI (on OR scale) for term if present, else None."""
        if name not in params.index:
            return None
        coef = float(params.loc[name])
        se = float(bse.loc[name]) if (bse is not None and name in bse.index) else None
        z = float(coef / se) if (se is not None) else None
        p = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
        or_val = float(np.exp(coef))
        # compute 95% CI on log-odds then exponentiate
        if se is not None:
            lo = coef - 1.96 * se
            hi = coef + 1.96 * se
            or_lo = float(np.exp(lo))
            or_hi = float(np.exp(hi))
        else:
            or_lo = or_hi = None
        return {
            'coef': coef,
            'std_err': se,
            'z': z,
            'p_value': p,
            'odds_ratio': or_val,
            'odds_ratio_95ci': (or_lo, or_hi)
        }

    # Identify parameter names robustly.
    idx = list(params.index)

    # likely names
    name_rel = None
    name_dist = None
    name_inter = None

    # exact matches first
    if 'RelSize_z' in idx:
        name_rel = 'RelSize_z'
    if 'DistDiff_z' in idx:
        name_dist = 'DistDiff_z'

    # find interaction: typically 'RelSize_z:DistDiff_z' or 'DistDiff_z:RelSize_z'
    for nm in idx:
        if (('RelSize_z' in nm) and ('DistDiff_z' in nm) and (':' in nm or '*' in nm or '.' in nm or '/' in nm)):
            name_inter = nm
            break
    # fallback: if no ':' but some other form, try any name containing both substrings
    if name_inter is None:
        for nm in idx:
            if ('RelSize_z' in nm) and ('DistDiff_z' in nm) and nm != 'RelSize_z' and nm != 'DistDiff_z':
                name_inter = nm
                break

    # fallback names if not found by exact match: try to find by containment
    if name_rel is None:
        for nm in idx:
            if 'RelSize' in nm and 'RelSize_z' not in nm:
                name_rel = nm
                break
    if name_dist is None:
        for nm in idx:
            if 'DistDiff' in nm and 'DistDiff_z' not in nm:
                name_dist = nm
                break

    # collect main predictors
    results['RelSize'] = collect_term(name_rel) if name_rel else None
    results['DistDiff'] = collect_term(name_dist) if name_dist else None
    results['Interaction'] = collect_term(name_inter) if name_inter else None

    # Collect location dummies (any parameter name starting with "C(Location)" or similar)
    loc_terms = {nm: collect_term(nm) for nm in idx if ('C(Location)' in nm) or ('Location' in nm and 'C(Location)' not in nm and 'dyad' not in nm and ('T.' in nm or '[' in nm))}
    results['Location_terms'] = loc_terms

    # Simple slopes: effect of RelSize_z at DistDiff_z = -1, 0, +1
    simple_slopes = {}
    if name_rel:
        # coefficient names for RelSize and interaction must be present to compute slope at nonzero DistDiff
        b_rel = params.get(name_rel, np.nan)
        b_inter = params.get(name_inter, 0.0) if name_inter else 0.0
        # For slope se, we need var(b_rel) + d^2 var(b_inter) + 2 d cov(b_rel,b_inter)
        for d in [-1.0, 0.0, 1.0]:
            slope = float(b_rel + b_inter * d)
            slope_dict = {'slope_logodds': slope}
            # compute SE if cov available
            slope_se = None
            if cov_df is not None and name_rel in cov_df.index:
                var_rel = cov_df.loc[name_rel, name_rel]
                var_inter = cov_df.loc[name_inter, name_inter] if (name_inter in cov_df.index) else 0.0
                cov_rel_inter = cov_df.loc[name_rel, name_inter] if (name_inter in cov_df.index) else 0.0
                var_s = var_rel + (d ** 2) * var_inter + 2 * d * cov_rel_inter
                if var_s < 0:
                    # numerical issues; avoid negative var
                    var_s = max(var_s, 0.0)
                slope_se = float(np.sqrt(var_s))
                slope_dict['std_err'] = slope_se
                # 95% CI on log-odds
                lo = slope - 1.96 * slope_se
                hi = slope + 1.96 * slope_se
                slope_dict['odds_ratio'] = float(np.exp(slope))
                slope_dict['odds_ratio_95ci'] = (float(np.exp(lo)), float(np.exp(hi)))
            else:
                # fallback: if only bse for the RelSize exists (no cov), approximate using bse of RelSize (valid only for d=0)
                if (d == 0) and (bse is not None) and (name_rel in bse.index):
                    slope_se = float(bse.loc[name_rel])
                    slope_dict['std_err'] = slope_se
                    lo = slope - 1.96 * slope_se
                    hi = slope + 1.96 * slope_se
                    slope_dict['odds_ratio'] = float(np.exp(slope))
                    slope_dict['odds_ratio_95ci'] = (float(np.exp(lo)), float(np.exp(hi)))
                else:
                    slope_dict['std_err'] = None
                    slope_dict['odds_ratio'] = float(np.exp(slope))
                    slope_dict['odds_ratio_95ci'] = (None, None)
            simple_slopes[f'DistDiff_z={d}'] = slope_dict
    results['Simple_slopes_RelSize_at_DistDiff'] = simple_slopes

    # Build a short textual interpretation
    lines = []
    # RelSize
    rel = results['RelSize']
    if rel is not None:
        sig = '(p < 0.05)' if rel['p_value'] is not None and rel['p_value'] < 0.05 else '(ns)'
        lines.append(f"Relative group size (RelSize_z): coef={rel['coef']:.3f}, SE={rel['std_err']:.3f}, p={rel['p_value']:.3g} -> OR={rel['odds_ratio']:.3f} {sig}.")
        if rel['p_value'] is not None and rel['p_value'] < 0.05:
            lines.append("  Interpretation: Larger focal groups are more likely to win; a 1-SD increase in relative size multiplies the odds of winning by the OR above.")
    else:
        lines.append("Relative group size (RelSize_z) not found in model output.")

    # DistDiff
    dist = results['DistDiff']
    if dist is not None:
        sig = '(p < 0.05)' if dist['p_value'] is not None and dist['p_value'] < 0.05 else '(ns)'
        lines.append(f"Distance difference (DistDiff_z: contest closer to focal = positive): coef={dist['coef']:.3f}, SE={dist['std_err']:.3f}, p={dist['p_value']:.3g} -> OR={dist['odds_ratio']:.3f} {sig}.")
        if dist['p_value'] is not None and dist['p_value'] < 0.05:
            lines.append("  Interpretation: Contests closer to the focal group's center (positive DistDiff_z) increase the focal group's probability of winning.")
    else:
        lines.append("Distance difference (DistDiff_z) not found in model output.")

    # Interaction
    inter = results['Interaction']
    if inter is not None:
        sig = '(p < 0.05)' if inter['p_value'] is not None and inter['p_value'] < 0.05 else '(ns)'
        lines.append(f"Interaction (RelSize_z x DistDiff_z): coef={inter['coef']:.3f}, SE={inter['std_err']:.3f}, p={inter['p_value']:.3g} {sig}.")
        if inter['p_value'] is not None and inter['p_value'] < 0.05:
            lines.append("  Interpretation: The effect of relative group size on winning depends on contest location (the slope of RelSize_z changes with DistDiff_z).")
            # include simple slopes
            for key, val in results['Simple_slopes_RelSize_at_DistDiff'].items():
                orv = val.get('odds_ratio')
                ci = val.get('odds_ratio_95ci')
                if ci[0] is not None:
                    lines.append(f"   - At {key}: OR={orv:.3f}, 95%CI=({ci[0]:.3f}, {ci[1]:.3f}).")
                else:
                    lines.append(f"   - At {key}: OR={orv:.3f} (CI unavailable).")
        else:
            lines.append("  Interpretation: No strong evidence that the effect of relative size depends on contest location (interaction not statistically significant).")
            # still provide simple slopes (useful even if ns)
            for key, val in results['Simple_slopes_RelSize_at_DistDiff'].items():
                orv = val.get('odds_ratio')
                ci = val.get('odds_ratio_95ci')
                if ci[0] is not None:
                    lines.append(f"   - At {key}: OR={orv:.3f}, 95%CI=({ci[0]:.3f}, {ci[1]:.3f}).")
                else:
                    lines.append(f"   - At {key}: OR={orv:.3f} (CI unavailable).")
    else:
        lines.append("Interaction term not found in model output.")

    # Location dummies
    if results['Location_terms']:
        lines.append("Location dummy coefficients (contrasts vs. reference level):")
        for nm, t in results['Location_terms'].items():
            if t is None:
                continue
            sig = '(p < 0.05)' if (t['p_value'] is not None and t['p_value'] < 0.05) else '(ns)'
            lines.append(f"  {nm}: coef={t['coef']:.3f}, SE={t['std_err']:.3f}, p={t['p_value']:.3g} -> OR={t['odds_ratio']:.3f} {sig}")
        lines.append("  Interpretation: These coefficients are contrasts relative to the (omitted) reference Location level; positive coef -> higher odds of focal winning compared to reference.")
    else:
        lines.append("No Location dummy terms were identified in the model output.")

    # Join description
    description = " ".join(lines)

    return {
        "object": results,
        "description": description
    }