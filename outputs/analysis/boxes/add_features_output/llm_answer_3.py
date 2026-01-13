def extract_final_answer(model_output):
    """
    Extract estimates of the developmental slope (effect of age_c) on choosing the majority
    for each cultural site from a fitted statsmodels Logit results object that was fit with:
        MajorityChoice ~ age_c * C(culture) + ...
    
    Returns a dict with:
      - "object": pandas.DataFrame with rows per culture and columns:
            slope (log-odds change per 1 SD age),
            se, z, p, ci_lower, ci_upper
      - "description": string describing what the table means and whether cluster-robust
            covariances were used.
    
    The function uses a clustered covariance matrix if attached to the results object
    as `cov_cluster` (and associated bse_cluster), otherwise it falls back to the
    model's covariance matrix (cov_params()).
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object (missing .params).")

    params = model_output.params
    param_index = params.index.astype(str)

    # Determine which covariance matrix to use (cluster-robust if available)
    if hasattr(model_output, "cov_cluster"):
        cov = model_output.cov_cluster
        cov_source = "cluster-robust (model_output.cov_cluster)"
    else:
        try:
            cov = model_output.cov_params()
            cov_source = "model-based (cov_params())"
        except Exception:
            raise ValueError("Could not obtain covariance matrix from model_output.")

    # Ensure 'age_c' exists as a parameter
    if 'age_c' not in param_index:
        raise ValueError("The fitted model does not contain a parameter named 'age_c'. "
                         "Check that the model formula included age_c and that parameter names are unchanged.")

    # Get list of cultures from the original data used by the model (preserve order)
    try:
        df_orig = model_output.model.data.frame
        cultures = list(pd.Series(df_orig['culture'].values).astype(str).unique())
    except Exception:
        # Fallback: infer cultures from parameter names (those appearing in C(culture))
        cultures = []
        for pn in param_index:
            if pn.startswith('C(culture)[T.'):
                # extract the level name
                lvl = pn.split('C(culture)[T.')[1].rstrip(']')
                cultures.append(lvl)
        cultures = list(dict.fromkeys(cultures))  # unique preserve order
        # If we only recovered non-reference levels, we still need one reference slot name placeholder.
        # We'll try to infer ref by comparing to params later.

    if len(cultures) == 0:
        raise ValueError("Could not determine culture levels from model output.")

    # Identify the reference culture: the one with no corresponding C(culture)[T.<level>] param
    ref_culture = None
    for c in cultures:
        pname = f"C(culture)[T.{c}]"
        if pname not in param_index:
            ref_culture = c
            break
    # If none found among cultures (e.g., cultures list contained only non-reference ones),
    # try to infer any param-level names and pick the first culture not present as a param name.
    if ref_culture is None:
        # recover all unique levels from parameter names
        param_levels = []
        for pn in param_index:
            if pn.startswith('C(culture)[T.'):
                lvl = pn.split('C(culture)[T.')[1].rstrip(']')
                param_levels.append(lvl)
        # try to find a culture in `cultures` not in param_levels
        for c in cultures:
            if c not in param_levels:
                ref_culture = c
                break
    # as a last resort, pick the first culture as reference (but warn)
    if ref_culture is None:
        ref_culture = cultures[0]

    # Build results rows: for each culture, slope = coef(age_c) + coef(age_c:C(culture)[T.<level>]) (if present)
    rows = []
    for c in cultures:
        # contrast vector 'a' of same length/order as params
        a = np.zeros(len(params), dtype=float)
        # index position helper
        name_to_idx = {n: i for i, n in enumerate(param_index)}

        # always include the main 'age_c' term
        a[name_to_idx['age_c']] = 1.0

        # if culture is not the reference, add the interaction term if present
        if c != ref_culture:
            inter_name = f'age_c:C(culture)[T.{c}]'
            # There is a small possibility statsmodels names the interaction differently,
            # also try the alternative ordering 'age_c:C(culture)[T.{c}]' vs 'age_c:C(culture)[T.{c}]' (same)
            if inter_name in name_to_idx:
                a[name_to_idx[inter_name]] = 1.0
            else:
                # Try alternative naming convention: 'age_c:C(culture)[T.<level>]'
                alt_candidates = [pn for pn in param_index if pn.startswith('age_c:') and f"[T.{c}]" in pn]
                if len(alt_candidates) == 1:
                    a[name_to_idx[alt_candidates[0]]] = 1.0
                # else, if not found, assume no interaction coefficient (i.e., treat as 0)
        # compute estimate and its variance
        est = float(a.dot(params.values))
        var = float(a.dot(cov).dot(a))
        se = float(np.sqrt(var)) if var >= 0 else np.nan
        z = est / se if se and not np.isnan(se) else np.nan
        p = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
        ci_low = est - 1.96 * se if not np.isnan(se) else np.nan
        ci_high = est + 1.96 * se if not np.isnan(se) else np.nan

        rows.append({
            'culture': c,
            'is_reference': (c == ref_culture),
            'slope_logodds_per_1SD_age': est,
            'se': se,
            'z': z,
            'p': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high
        })

    df_res = pd.DataFrame(rows).set_index('culture')

    description = (
        "This table shows the estimated developmental slope (change in log-odds of choosing the "
        "majority per 1 SD increase in age) for each cultural site. Slopes equal the coefficient on "
        "'age_c' for the reference culture and 'age_c' + interaction coefficient for non-reference "
        "cultures. SE, z, p, and 95% CI are computed using the covariance matrix from: "
        f"{cov_source}. Positive slopes indicate increased reliance on the majority with age; "
        "negative slopes indicate decreased reliance with age. P-values are two-sided (normal approximation)."
    )

    return {"object": df_res, "description": description}