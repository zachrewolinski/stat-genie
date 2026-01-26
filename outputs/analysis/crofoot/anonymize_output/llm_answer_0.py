def extract_final_answer(model_output):
    """
    Extract coefficients, SEs, CIs, and test the effect of RelativeSize overall and by ContestLocation
    from the RobustResults-like object returned by the modeling function.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coefficients, standard errors, z, p, 95% CI,
                  odds ratios, and location-specific slopes for RelativeSize)
      - "description": brief human-readable interpretation focused on whether relative group size
                       and contest location influence the probability that the focal group wins.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats as sps

    res = model_output

    # Helper to obtain parameter names if res.params is an array without index
    def extract_param_names(result_obj, length):
        # common places to look for names in statsmodels-like objects
        if hasattr(result_obj, "model"):
            model = getattr(result_obj, "model")
            # many statsmodels objects expose exog_names on the model
            if hasattr(model, "exog_names"):
                try:
                    names = list(model.exog_names)
                    if len(names) == length:
                        return names
                except Exception:
                    pass
            # sometimes model has term_names or param_names
            if hasattr(model, "term_names"):
                try:
                    names = list(model.term_names)
                    if len(names) == length:
                        return names
                except Exception:
                    pass
        # result-level alternatives
        for attr in ("param_names", "params_names", "names", "columns"):
            if hasattr(result_obj, attr):
                try:
                    names = list(getattr(result_obj, attr))
                    if len(names) == length:
                        return names
                except Exception:
                    pass
        # If bse or pvalues are Series, use their index
        if hasattr(result_obj, "bse") and isinstance(result_obj.bse, (pd.Series, dict)):
            try:
                names = list(pd.Series(result_obj.bse).index)
                if len(names) == length:
                    return names
            except Exception:
                pass
        if hasattr(result_obj, "pvalues") and isinstance(result_obj.pvalues, (pd.Series, dict)):
            try:
                names = list(pd.Series(result_obj.pvalues).index)
                if len(names) == length:
                    return names
            except Exception:
                pass
        # Fallback generic names
        return [f"param_{i}" for i in range(length)]

    # Make params, bse, pvals into pandas Series with aligned indices (names)
    raw_params = getattr(res, "params", None)
    if raw_params is None:
        raise RuntimeError("model_output has no attribute 'params'")

    # Convert to list/array for length
    try:
        raw_params_arr = list(raw_params) if not isinstance(raw_params, pd.Series) else list(raw_params.values)
    except Exception:
        # try to coerce to numpy array then list
        raw_params_arr = list(np.asarray(raw_params))

    param_len = len(raw_params_arr)
    param_names = None

    # If params already a Series with meaningful index, keep it
    if isinstance(raw_params, pd.Series) and not isinstance(raw_params.index, pd.RangeIndex):
        params = raw_params.copy()
        param_names = list(params.index)
    else:
        # attempt to discover names
        names = extract_param_names(res, param_len)
        params = pd.Series(raw_params_arr, index=names)
        param_names = names

    # bse
    raw_bse = getattr(res, "bse", None)
    if raw_bse is None:
        # create NaNs
        bse = pd.Series([np.nan] * param_len, index=param_names)
    elif isinstance(raw_bse, pd.Series) and not isinstance(raw_bse.index, pd.RangeIndex):
        bse = raw_bse.reindex(param_names).astype(float)
    else:
        try:
            bse_arr = list(raw_bse)
        except Exception:
            bse_arr = list(np.asarray(raw_bse))
        bse = pd.Series(bse_arr, index=param_names).astype(float)

    # pvalues
    raw_pvals = getattr(res, "pvalues", None)
    if raw_pvals is None:
        pvals = pd.Series([np.nan] * param_len, index=param_names)
    elif isinstance(raw_pvals, pd.Series) and not isinstance(raw_pvals.index, pd.RangeIndex):
        pvals = raw_pvals.reindex(param_names).astype(float)
    else:
        try:
            pvals_arr = list(raw_pvals)
        except Exception:
            pvals_arr = list(np.asarray(raw_pvals))
        pvals = pd.Series(pvals_arr, index=param_names).astype(float)

    # confidence intervals
    try:
        ci_raw = getattr(res, "conf_int", None)
        if ci_raw is None:
            raise Exception("no conf_int")
        # if already DataFrame with index
        if isinstance(ci_raw, pd.DataFrame) and list(ci_raw.columns) and len(ci_raw.columns) >= 2:
            # try to pick first two columns as lower/upper
            ci_df = ci_raw.iloc[:, :2].copy()
            ci_df.columns = ['ci_lower', 'ci_upper']
            # reindex to param_names if needed
            if not isinstance(ci_df.index, pd.Index) or list(ci_df.index) != param_names:
                try:
                    ci_df = ci_df.reindex(param_names)
                except Exception:
                    ci_df.index = param_names
        else:
            # assume array-like shape (n_params,2)
            ci_arr = list(ci_raw)
            ci_df = pd.DataFrame(ci_arr, columns=['ci_lower', 'ci_upper'], index=param_names)
    except Exception:
        # compute from params +/- 1.96*bse
        crit = sps.norm.ppf(0.975)
        ci_df = pd.DataFrame({
            'ci_lower': params - crit * bse,
            'ci_upper': params + crit * bse
        }, index=param_names)

    # Covariance matrix for linear combinations (if available)
    cov_mat = None
    try:
        cov_raw = getattr(res, "cov_params", None)
        if cov_raw is None:
            cov_raw = getattr(res, "cov", None)
        if cov_raw is not None:
            # if it's a DataFrame
            if isinstance(cov_raw, pd.DataFrame):
                # try to reindex to param_names
                try:
                    cov_mat = np.asarray(cov_raw.loc[param_names, param_names])
                except Exception:
                    cov_mat = np.asarray(cov_raw)
            else:
                cov_mat = np.asarray(cov_raw)
            if cov_mat.shape[0] != len(param_names) or cov_mat.shape[1] != len(param_names):
                # shape mismatch; ignore cov_mat
                cov_mat = None
    except Exception:
        cov_mat = None

    # Helper to safely get param stats
    def get_param_stats(name):
        if name not in params.index:
            return None
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index and not pd.isna(bse[name]) else float(np.nan)
        z = float(coef / se) if se != 0 and not np.isnan(se) else np.nan
        p = float(pvals[name]) if name in pvals.index and not pd.isna(pvals[name]) else (float(2 * (1 - sps.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan)
        ci_low = float(ci_df.loc[name, 'ci_lower']) if name in ci_df.index else float(np.nan)
        ci_high = float(ci_df.loc[name, 'ci_upper']) if name in ci_df.index else float(np.nan)
        or_ = float(np.exp(coef)) if not np.isnan(coef) else float(np.nan)
        try:
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        except Exception:
            or_ci = (float(np.nan), float(np.nan))
        return {
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'odds_ratio': or_,
            'odds_ratio_ci': or_ci
        }

    param_names = list(params.index)

    # Identify parameter names for location main effects and interactions.
    # Expected names in this model: 
    #   'RelativeSize'
    #   'C(ContestLocation)[T.FocalHome]' and 'C(ContestLocation)[T.OtherHome]' (main effects)
    #   'RelativeSize:C(ContestLocation)[T.FocalHome]' and 'RelativeSize:C(ContestLocation)[T.OtherHome]' (interactions)
    results = {'params': {}, 'relative_size_location_slopes': {}}

    # Find the RelativeSize parameter name robustly
    rel_name = None
    for n in param_names:
        if n == 'RelativeSize' or n.endswith('.RelativeSize') or n.endswith(':RelativeSize') or n.endswith('RelativeSize'):
            rel_name = n
            break
    # try case-insensitive match as fallback
    if rel_name is None:
        for n in param_names:
            if str(n).lower() == 'relativesize' or 'relativesize' in str(n).lower().split(':') or str(n).lower().endswith('relativesize'):
                rel_name = n
                break

    if rel_name is None:
        raise RuntimeError("Could not find a parameter named 'RelativeSize' in model output.")

    results['params'][rel_name] = get_param_stats(rel_name)

    # get location main effect names
    loc_levels = ['FocalHome', 'OtherHome']
    loc_main_names = {}
    loc_inter_names = {}
    for lvl in loc_levels:
        main_name = f"C(ContestLocation)[T.{lvl}]"
        inter_name = f"RelativeSize:C(ContestLocation)[T.{lvl}]"
        alt_inter_name = f"C(ContestLocation)[T.{lvl}]:RelativeSize"

        # find main
        if main_name in param_names:
            loc_main_names[lvl] = main_name
        else:
            found = [n for n in param_names if ("C(ContestLocation)" in str(n) and lvl in str(n) and ':' not in str(n))]
            loc_main_names[lvl] = found[0] if found else None

        # find interaction
        if inter_name in param_names:
            loc_inter_names[lvl] = inter_name
        elif alt_inter_name in param_names:
            loc_inter_names[lvl] = alt_inter_name
        else:
            found = [n for n in param_names if ('RelativeSize' in str(n) and lvl in str(n))]
            loc_inter_names[lvl] = found[0] if found else None

        # store main and interaction stats if present
        if loc_main_names[lvl] is not None:
            results['params'][loc_main_names[lvl]] = get_param_stats(loc_main_names[lvl])
        if loc_inter_names[lvl] is not None:
            results['params'][loc_inter_names[lvl]] = get_param_stats(loc_inter_names[lvl])

    # Baseline location is 'Neutral' (by modeling note). Compute the slope (log-odds change per unit RelativeSize)
    # for each location: Neutral, FocalHome, OtherHome. Slope = coef(RelativeSize) + coef(interaction if present)
    def compute_slope_for_location(loc):
        # weights dict param_name -> weight
        weights = {}
        weights[rel_name] = 1.0
        inter = loc_inter_names.get(loc)
        if inter and inter in params.index:
            weights[inter] = 1.0

        # compute coef
        coef = sum(float(params[p]) * w for p, w in weights.items())

        # compute standard error using cov matrix if available, else approximate by sqrt(sum(se^2))
        if cov_mat is not None:
            # Build weight vector in order of param_names
            wvec = np.zeros(len(param_names))
            name_to_idx = {n: i for i, n in enumerate(param_names)}
            for p, w in weights.items():
                if p in name_to_idx:
                    wvec[name_to_idx[p]] = w
            try:
                var = float(wvec @ cov_mat @ wvec)
                se = float(np.sqrt(var)) if var >= 0 else float(np.nan)
            except Exception:
                se = float(np.nan)
        else:
            # approximate using independence assumption
            se = float(np.sqrt(sum((float(bse[p]) * w) ** 2 for p, w in weights.items() if p in bse.index and not pd.isna(bse[p])))) if any((p in bse.index and not pd.isna(bse[p])) for p in weights) else float(np.nan)

        z = float(coef / se) if se != 0 and not np.isnan(se) else np.nan
        p = float(2 * (1 - sps.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
        crit = sps.norm.ppf(0.975)
        ci_low = float(coef - crit * se) if not np.isnan(se) else float(np.nan)
        ci_high = float(coef + crit * se) if not np.isnan(se) else float(np.nan)
        or_ = float(np.exp(coef)) if not np.isnan(coef) else float(np.nan)
        try:
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        except Exception:
            or_ci = (float(np.nan), float(np.nan))

        return {
            'location': loc,
            'slope_coef': coef,
            'slope_se': se,
            'slope_z': z,
            'slope_p': p,
            'slope_ci_lower': ci_low,
            'slope_ci_upper': ci_high,
            'slope_odds_ratio_per_unit': or_,
            'slope_or_ci': or_ci
        }

    # Compute for Neutral (baseline)
    results['relative_size_location_slopes']['Neutral'] = compute_slope_for_location('Neutral')

    # Compute for each level
    for lvl in loc_levels:
        results['relative_size_location_slopes'][lvl] = compute_slope_for_location(lvl)

    # Create concise interpretation
    def interpret_slope(s):
        if np.isnan(s.get('slope_p', np.nan)):
            return "slope or p-value unavailable"
        sign = "positive" if s['slope_coef'] > 0 else ("negative" if s['slope_coef'] < 0 else "null")
        sig = "statistically significant (p < 0.05)" if s['slope_p'] < 0.05 else f"not statistically significant (p = {s['slope_p']:.3f})"
        return f"{sign} effect; {sig}"

    interp_lines = []
    # Overall (Neutral baseline) statement
    neutral = results['relative_size_location_slopes']['Neutral']
    interp_lines.append(f"Neutral (baseline) — RelativeSize slope: {neutral['slope_coef']:.3f}, SE={neutral['slope_se']:.3f}, p={neutral['slope_p']:.3f}. Interpretation: {interpret_slope(neutral)}.")
    for lvl in loc_levels:
        s = results['relative_size_location_slopes'][lvl]
        interp_lines.append(f"{lvl} — RelativeSize slope: {s['slope_coef']:.3f}, SE={s['slope_se']:.3f}, p={s['slope_p']:.3f}. Interpretation: {interpret_slope(s)}.")

    # Test for "home advantage" in terms of slope magnitude: compare slope at FocalHome vs OtherHome and Neutral
    def compare_slopes(a, b):
        # simple numeric comparison of coef sizes; check whether difference's CI excludes 0 if cov matrix available
        diff = a['slope_coef'] - b['slope_coef']
        if cov_mat is not None:
            # Compute variance of difference via weight vectors
            name_to_idx = {n: i for i, n in enumerate(param_names)}
            def build_w(loc):
                w = np.zeros(len(param_names))
                if rel_name in name_to_idx:
                    w[name_to_idx[rel_name]] = 1.0
                inter = loc_inter_names.get(loc)
                if inter and inter in name_to_idx:
                    w[name_to_idx[inter]] = 1.0
                return w
            w_a = build_w(a['location'] if 'location' in a else a)
            w_b = build_w(b['location'] if 'location' in b else b)
            w_diff = w_a - w_b
            try:
                var_diff = float(w_diff @ cov_mat @ w_diff)
                se_diff = np.sqrt(var_diff) if var_diff >= 0 else np.nan
                z = diff / se_diff if se_diff != 0 else np.nan
                p = 2 * (1 - sps.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                return {'diff': diff, 'se_diff': se_diff, 'z': z, 'p': p}
            except Exception:
                return {'diff': diff, 'se_diff': None, 'z': None, 'p': None}
        else:
            return {'diff': diff, 'se_diff': None, 'z': None, 'p': None}

    comp_focal_vs_other = compare_slopes(results['relative_size_location_slopes']['FocalHome'],
                                         results['relative_size_location_slopes']['OtherHome'])
    if comp_focal_vs_other.get('p') is not None:
        if comp_focal_vs_other['p'] < 0.05:
            comp_phrase = f"Slope for RelativeSize is larger at FocalHome than OtherHome (difference={comp_focal_vs_other['diff']:.3f}, p={comp_focal_vs_other['p']:.3f}), consistent with a home advantage."
        else:
            comp_phrase = f"No statistically significant difference in the RelativeSize slope between FocalHome and OtherHome (difference={comp_focal_vs_other['diff']:.3f}, p={comp_focal_vs_other['p']:.3f})."
    else:
        # fallback numeric comparison
        diff = comp_focal_vs_other['diff']
        if abs(diff) < 1e-6:
            comp_phrase = "Slopes in FocalHome and OtherHome are numerically identical (no interaction detected)."
        elif diff > 0:
            comp_phrase = "Slope is numerically larger in FocalHome than OtherHome (possible home advantage), but no formal test available."
        else:
            comp_phrase = "Slope is numerically smaller in FocalHome than OtherHome (no focal home advantage), but no formal test available."

    interp_lines.append(comp_phrase)

    description = " ".join(interp_lines)

    return {
        "object": results,
        "description": description
    }