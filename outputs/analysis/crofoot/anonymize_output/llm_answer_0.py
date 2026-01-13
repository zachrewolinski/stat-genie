def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and odds-ratios
    for key terms in the fitted GLM (logistic) model returned by the
    modeling function. Also computes the effect of relative group size
    (RelSize_z) separately for each location level when interaction terms
    RelSize_z:C(Location)[T.<level>] are present.
    
    Returns:
      {
        "object": {
            "terms": { term_name: {coef, se, p, ci_low, ci_high, OR, OR_ci_low, OR_ci_high}, ...},
            "rel_size_by_location": { location_level: {coef, se, p_approx, ci_low, ci_high, OR, OR_ci_low, OR_ci_high}, ...},
            "baseline_location": <baseline level name or None>,
            "notes": <any warnings or fallbacks used>
        },
        "description": <brief human-readable interpretation>
      }
    """
    import re
    import numpy as np
    import pandas as pd

    res = model_output  # shorthand

    # Prepare containers
    notes = []
    term_summaries = {}

    # Try to get parameter estimates, SEs, p-values, and conf-int
    try:
        params = res.params.copy()
    except Exception as e:
        raise ValueError(f"Could not extract params from model_output: {e}")

    # Some wrappers store pvalues under different attributes; attempt common ones
    try:
        bse = res.bse.copy()
    except Exception:
        # fallback: derive from cov_params diag if available
        try:
            cov = res.cov_params()
            bse = np.sqrt(np.diag(cov))
            bse = pd.Series(bse, index=params.index)
            notes.append("bse not available; computed from cov_params diagonal.")
        except Exception:
            raise ValueError("Could not obtain standard errors (bse) from model_output.")

    try:
        pvalues = res.pvalues.copy()
    except Exception:
        pvalues = pd.Series(index=params.index, data=[np.nan]*len(params))
        notes.append("p-values not available in model_output; set to NaN.")

    # Confidence intervals
    try:
        ci = res.conf_int()
        # conf_int returns DataFrame with 2 columns; ensure correct indexing
        if isinstance(ci, np.ndarray):
            ci = pd.DataFrame(ci, index=params.index, columns=['2.5%', '97.5%'])
        else:
            ci.columns = ['2.5%', '97.5%']
    except Exception:
        # fallback using normal approximation
        ci_low = params - 1.96 * bse
        ci_high = params + 1.96 * bse
        ci = pd.DataFrame({'2.5%': ci_low, '97.5%': ci_high}, index=params.index)
        notes.append("conf_int not available; used Wald normal approximation (±1.96*SE).")

    # Odds ratios and OR CIs
    OR = np.exp(params)
    OR_ci_low = np.exp(ci['2.5%'])
    OR_ci_high = np.exp(ci['97.5%'])

    # Build term summaries for all parameters, but we will highlight key ones below
    for name in params.index:
        term_summaries[name] = {
            'coef': float(params[name]),
            'se': float(bse[name]) if name in bse.index else None,
            'p': float(pvalues[name]) if name in pvalues.index else None,
            'ci_2.5%': float(ci.loc[name, '2.5%']),
            'ci_97.5%': float(ci.loc[name, '97.5%']),
            'OR': float(OR[name]),
            'OR_ci_low': float(OR_ci_low[name]),
            'OR_ci_high': float(OR_ci_high[name]),
        }

    # Identify main terms of interest
    # Main continuous predictors: RelSize_z, DistanceDiff_z, MaleAdvantage
    main_terms = {}
    for t in ['RelSize_z', 'DistanceDiff_z', 'MaleAdvantage']:
        if t in params.index:
            main_terms[t] = term_summaries[t]
        else:
            notes.append(f"Term '{t}' not found in model parameters.")

    # Identify categorical Location dummy terms and interactions
    location_terms = [n for n in params.index if re.match(r'^C\(Location\)\[T\..+\]$', n)]
    interaction_terms = [n for n in params.index if ('RelSize_z' in n and 'C(Location)' in n) or re.match(r'^RelSize_z:C\(Location\)\[T\..+\]$', n) or re.match(r'^RelSize_z:C\(Location\)\[T\..+\]$', n)]
    # Also accept patterns where colon order might be reversed
    if not interaction_terms:
        interaction_terms = [n for n in params.index if (':C(Location)' in n and 'RelSize_z' in n) or (':RelSize_z' in n and 'C(Location)' in n)]

    # Extract location level names from the parameter names (e.g., C(Location)[T.FocalRange] -> FocalRange)
    location_levels_in_params = []
    for name in location_terms:
        m = re.search(r'C\(Location\)\[T\.(.+)\]', name)
        if m:
            location_levels_in_params.append(m.group(1))

    # Try to get full set of Location categories from the model data if possible to identify baseline
    baseline = None
    all_location_levels = None
    try:
        data_df = res.model.data.frame
        if 'Location' in data_df.columns:
            if pd.api.types.is_categorical_dtype(data_df['Location']):
                all_location_levels = list(data_df['Location'].cat.categories)
            else:
                all_location_levels = sorted(list(pd.unique(data_df['Location'])))
    except Exception:
        # unable to access original data frame; infer only from params
        pass

    if all_location_levels is not None:
        # baseline is the level in all_location_levels that's not present as a dummy in params
        baseline_candidates = [lvl for lvl in all_location_levels if f"C(Location)[T.{lvl}]" not in params.index]
        baseline = baseline_candidates[0] if baseline_candidates else None
    else:
        # infer baseline from param names if possible (if exactly 2 location dummies present, baseline is the third unknown)
        if len(location_levels_in_params) > 0:
            baseline = None  # we can't be sure without the original categories
            notes.append("Could not retrieve categorical ordering from model data; baseline location level unknown.")
        else:
            baseline = None

    # Prepare relsize effect by location:
    rel_size_by_location = {}
    if 'RelSize_z' not in params.index:
        notes.append("RelSize_z main effect not present; cannot compute effects by location.")
    else:
        coef_rel = float(params['RelSize_z'])
        se_rel = float(bse['RelSize_z']) if 'RelSize_z' in bse.index else None
        p_rel = float(pvalues['RelSize_z']) if 'RelSize_z' in pvalues.index else None

        # gather interaction coefficients keyed by level
        inter_by_level = {}
        for name in params.index:
            # patterns like 'RelSize_z:C(Location)[T.FocalRange]' or 'RelSize_z:C(Location)[T.OtherRange]'
            m = re.search(r'RelSize_z(?::|:)C\(Location\)\[T\.(.+)\]', name)
            if not m:
                m = re.search(r'RelSize_z:C\(Location\)\[T\.(.+)\]', name)
            if m:
                lvl = m.group(1)
                inter_by_level[lvl] = name

        # For baseline level, effect is the main effect
        if baseline is not None:
            lvlname = baseline
            coef = coef_rel
            se = se_rel
            # p-value approximate: use main effect p-value
            p_approx = p_rel
            # CI: use main effect CI
            ci_low = float(ci.loc['RelSize_z', '2.5%'])
            ci_high = float(ci.loc['RelSize_z', '97.5%'])
            OR_val = float(np.exp(coef))
            OR_ci_l = float(np.exp(ci_low))
            OR_ci_h = float(np.exp(ci_high))
            rel_size_by_location[lvlname] = {
                'coef': coef, 'se': se, 'p_approx': p_approx,
                'ci_2.5%': ci_low, 'ci_97.5%': ci_high,
                'OR': OR_val, 'OR_ci_low': OR_ci_l, 'OR_ci_high': OR_ci_h,
                'note': 'baseline (no location dummy)'
            }
        else:
            # If baseline unknown, still provide "baseline" as 'reference' using the main effect
            rel_size_by_location['(reference level)'] = {
                'coef': coef_rel, 'se': se_rel, 'p_approx': p_rel,
                'ci_2.5%': float(ci.loc['RelSize_z', '2.5%']),
                'ci_97.5%': float(ci.loc['RelSize_z', '97.5%']),
                'OR': float(np.exp(coef_rel)),
                'OR_ci_low': float(np.exp(ci.loc['RelSize_z', '2.5%'])),
                'OR_ci_high': float(np.exp(ci.loc['RelSize_z', '97.5%'])),
                'note': 'main-effect (applies to the reference location level)'
            }

        # For each non-baseline level present in interaction terms, compute combined effect
        # Need covariance matrix to compute SE of sum; fall back to ignoring covariance if not available.
        covmat = None
        try:
            covmat = res.cov_params()
        except Exception:
            notes.append("Could not obtain full covariance matrix; SE for combined effects will use sqrt(se_main^2 + se_inter^2) (ignoring covariance).")

        for lvl, pname in inter_by_level.items():
            coef_inter = float(params[pname])
            se_inter = float(bse[pname]) if pname in bse.index else None
            # combined coef = main + interaction
            combined_coef = coef_rel + coef_inter

            # compute SE of combined:
            se_combined = None
            if covmat is not None and ('RelSize_z' in covmat.index) and (pname in covmat.index):
                cov_rr = covmat.loc['RelSize_z', 'RelSize_z'] if 'RelSize_z' in covmat.index else None
                cov_ri = covmat.loc[pname, pname] if pname in covmat.index else None
                cov_cross = covmat.loc['RelSize_z', pname]
                # variance = var(main) + var(inter) + 2*cov(main,inter)
                var_comb = cov_rr + cov_ri + 2 * cov_cross
                se_combined = float(np.sqrt(var_comb)) if var_comb >= 0 else float(np.nan)
            else:
                # fallback
                if (se_rel is not None) and (se_inter is not None):
                    se_combined = float(np.sqrt(se_rel**2 + se_inter**2))
                else:
                    se_combined = None

            # approximate p-value using z-test if se_combined available
            if se_combined and se_combined > 0:
                z = combined_coef / se_combined
                # two-sided p-value
                from scipy import stats
                p_comb = float(2 * (1 - stats.norm.cdf(abs(z))))
                ci_low = combined_coef - 1.96 * se_combined
                ci_high = combined_coef + 1.96 * se_combined
            else:
                p_comb = None
                ci_low = None
                ci_high = None

            OR_c = float(np.exp(combined_coef))
            OR_c_low = float(np.exp(ci_low)) if ci_low is not None else None
            OR_c_high = float(np.exp(ci_high)) if ci_high is not None else None

            rel_size_by_location[lvl] = {
                'coef': combined_coef,
                'se': se_combined,
                'p_approx': p_comb,
                'ci_2.5%': ci_low,
                'ci_97.5%': ci_high,
                'OR': OR_c,
                'OR_ci_low': OR_c_low,
                'OR_ci_high': OR_c_high,
                'note': f"combined effect = main RelSize_z + interaction for Location={lvl}"
            }

    # Assemble final object
    final_object = {
        'terms': term_summaries,
        'main_terms_of_interest': main_terms,
        'location_dummies_in_params': location_terms,
        'interaction_terms_in_params': interaction_terms,
        'rel_size_by_location': rel_size_by_location,
        'baseline_location': baseline,
        'notes': notes
    }

    # Create a concise human-readable description
    # We'll base wording on RelSize_z main effect and its OR and p-value (if available).
    desc_lines = []
    if 'RelSize_z' in main_terms:
        m = main_terms['RelSize_z']
        p = m['p']
        orv = m['OR']
        coef = m['coef']
        sig = (p is not None) and (p < 0.05)
        desc_lines.append(
            f"Relative group size (RelSize_z): coefficient = {coef:.3f}, OR = {orv:.3f}, p = {p:.3g}."
            + (" This effect is statistically significant (p < 0.05)." if sig else " Not statistically significant at p < 0.05.")
        )
    else:
        desc_lines.append("RelSize_z main effect not found in model parameters.")

    if 'DistanceDiff_z' in main_terms:
        m = main_terms['DistanceDiff_z']
        desc_lines.append(
            f"Relative contest location (DistanceDiff_z): coefficient = {m['coef']:.3f}, OR = {m['OR']:.3f}, p = {m['p']:.3g}."
        )
    else:
        desc_lines.append("DistanceDiff_z main effect not found in model parameters.")

    # Note interactions if present
    if interaction_terms:
        desc_lines.append("Interactions between RelSize_z and Location are present; the effect of relative group size differs by contest location. See 'rel_size_by_location' for combined coefficients, SEs, approximate p-values, and ORs by location level.")
    else:
        desc_lines.append("No interaction terms between RelSize_z and Location detected; the effect of relative group size is modeled as constant across locations (aside from the continuous DistanceDiff_z term).")

    # Short interpretive summary about direction:
    # Use main RelSize_z coef sign to indicate direction
    if 'RelSize_z' in main_terms:
        coef = main_terms['RelSize_z']['coef']
        if coef > 0:
            desc_lines.append("Positive RelSize_z coefficient indicates that having more group members relative to the other group increases the probability that the focal group wins (OR > 1).")
        elif coef < 0:
            desc_lines.append("Negative RelSize_z coefficient indicates that having more group members relative to the other group decreases the probability that the focal group wins (OR < 1).")
        else:
            desc_lines.append("RelSize_z coefficient near zero indicates no effect of relative group size on winning probability.")
    # Compile description
    description = " ".join(desc_lines)

    return {"object": final_object, "description": description}