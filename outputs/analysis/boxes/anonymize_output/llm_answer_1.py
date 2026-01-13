def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear and quadratic) and site-specific age slopes
    from the fitted GLM model output dictionary produced by the provided `model(...)`
    function.

    Returns a dict with:
      - "object": a dict containing:
          - "age_sq": info about the Age_sq coefficient (beta, se, p, 95% CI)
          - "reference_site": age slope (log-odds per unit Age_c at mean age) for the
            model reference site (i.e., when no SiteID indicator is present)
          - "sites": mapping from each site (parsed from parameter names) to the
            age slope at mean age (log-odds), SE, z, p, odds-ratio and 95% CI,
            and which interaction parameter was used
          - "notes": any pragmatic notes about reference site naming or missing params
      - "description": a plain-language interpretation of the returned numbers:
          what the per-site slopes mean (change in log-odds / odds ratio per unit
          increase in mean-centered age), and what the Age_sq coefficient implies.

    The function is defensive: it prefers clustered robust results if present,
    falls back to the main fitted result, and returns a clear message if no model
    is available.
    """
    import numpy as np
    import re

    # Prefer clustered robust results if available
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('glm_result_clustered_se') or model_output.get('glm_result')
    else:
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No fitted model object found in model_output (both 'glm_result' and 'glm_result_clustered_se' are None)."
        }

    # Safely get params, covariance, pvalues, conf_int
    try:
        params = res.params
    except Exception as e:
        return {"object": None, "description": f"Could not access params from model result: {e}"}
    try:
        cov = res.cov_params()
    except Exception:
        # Ensure cov is available as DataFrame-like; if not, try to build from bse (less ideal)
        cov = None
    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = None
    try:
        conf = res.conf_int()
    except Exception:
        conf = None

    # Ensure index is string-list friendly
    param_names = [str(n) for n in params.index]

    # Helper to find parameter by substring heuristics
    def find_param_containing(sub):
        for n in param_names:
            if sub == n:
                return n
        for n in param_names:
            if sub in n:
                return n
        return None

    # Identify Age_c and Age_sq parameter names
    age_param = find_param_containing('Age_c')
    age_sq_param = find_param_containing('Age_sq')

    # Identify interaction parameters for Age_c x SiteID:
    # look for parameter names that contain both 'Age_c' and 'SiteID' (or 'C(SiteID)').
    interaction_params = [n for n in param_names if ('Age_c' in n) and ('SiteID' in n)]
    # Also allow the reversed order: 'SiteID' then 'Age_c'
    interaction_params += [n for n in param_names if ('SiteID' in n) and ('Age_c' in n) and n not in interaction_params]

    # Identify site main effect parameters (to parse site labels)
    site_main_params = [n for n in param_names if ('SiteID' in n) and ('Age_c' not in n)]

    # Parse site labels from parameter names using pattern [T.<site>] or :<site>
    def parse_site_label(param_name):
        # common statsmodels pattern: "C(SiteID)[T.site_name]" or "C(SiteID)[T.site]"
        m = re.search(r'\[T\.([^]]+)\]', param_name)
        if m:
            return m.group(1)
        # alternative pattern when colon used: "C(SiteID)[T.site]:Age_c" or "Age_c:C(SiteID)[T.site]"
        m = re.search(r'C\(SiteID\)\[T\.([^]]+)\]', param_name)
        if m:
            return m.group(1)
        # fallback: try anything after last '.' or last ':' that looks like a site token
        if '.' in param_name:
            return param_name.split('.')[-1]
        if ':' in param_name:
            return param_name.split(':')[-1]
        return param_name

    sites = []
    for n in site_main_params + interaction_params:
        label = parse_site_label(n)
        if label not in sites:
            sites.append(label)

    # Compute site-specific age slopes at mean age (Age_c = 0). Quadratic term contributes 0 at Age_c=0.
    # The reference (omitted) site is the one whose age slope equals the Age_c main effect alone.
    results_per_site = {}
    # Ensure cov is a DataFrame-like with .loc access; if not available, create approximate diagonal from res.bse^2
    cov_df = cov
    if cov_df is None:
        # Try to build diagonal covariance from bse if available
        try:
            bse = res.bse
            cov_diag = np.square(bse.values)
            cov_df = None
            # Build as a dict for lookup
            cov_lookup = {name: cov_diag[i] for i, name in enumerate(param_names)}
            def cov_get(a, b):
                if a == b:
                    return cov_lookup.get(a, np.nan)
                # covariance unknown -> assume 0 (conservative)
                return 0.0
        except Exception:
            cov_get = lambda a, b: np.nan
    else:
        # Ensure cov_df is indexed by param names strings
        try:
            # If it's a numpy array, convert with param_names
            import pandas as _pd
            if not hasattr(cov_df, 'loc'):
                cov_df = _pd.DataFrame(cov_df, index=param_names, columns=param_names)
            else:
                # ensure indices are strings
                cov_df.index = [str(i) for i in cov_df.index]
                cov_df.columns = [str(i) for i in cov_df.columns]
            def cov_get(a, b):
                try:
                    return float(cov_df.loc[a, b])
                except Exception:
                    return float(cov_df.loc[str(a), str(b)])
        except Exception:
            def cov_get(a, b):
                return np.nan

    # Helper normal p-value using z
    try:
        from scipy import stats
        def z_to_p(z):
            return 2.0 * (1 - stats.norm.cdf(abs(z)))
    except Exception:
        # Approximate using error function if scipy not available
        import math
        def z_to_p(z):
            # CDF of standard normal
            cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            return 2.0 * (1 - cdf) if z >= 0 else 2.0 * cdf

    if age_param is None:
        return {"object": None, "description": "Could not find an 'Age_c' parameter in the fitted model parameters."}

    beta_age = float(params[age_param])
    var_age = cov_get(age_param, age_param)
    for site in sites:
        # find interaction parameter name for this site if present
        inter_param = None
        for n in interaction_params:
            if site in n:
                inter_param = n
                break
        beta_inter = float(params[inter_param]) if (inter_param is not None and inter_param in params.index) else 0.0

        slope = beta_age + beta_inter  # derivative of log-odds at Age_c=0 (mean age)
        # variance = Var(beta_age) + Var(beta_inter) + 2*Cov(beta_age, beta_inter)
        try:
            var_inter = cov_get(inter_param, inter_param) if inter_param is not None else 0.0
            cov_ai = cov_get(age_param, inter_param) if inter_param is not None else 0.0
            var_slope = var_age + var_inter + 2.0 * cov_ai
            if var_slope < 0 and abs(var_slope) < 1e-12:
                var_slope = 0.0
            se_slope = float(np.sqrt(var_slope)) if (var_slope is not None and not np.isnan(var_slope)) else float('nan')
        except Exception:
            se_slope = float('nan')

        z = slope / se_slope if (se_slope and not np.isnan(se_slope) and se_slope > 0) else float('nan')
        p = float(z_to_p(z)) if not np.isnan(z) else float('nan')
        or_ratio = float(np.exp(slope))
        ci_low = float(np.exp(slope - 1.96 * se_slope)) if (not np.isnan(se_slope)) else None
        ci_high = float(np.exp(slope + 1.96 * se_slope)) if (not np.isnan(se_slope)) else None

        results_per_site[site] = {
            "interaction_param": inter_param,
            "slope_log_odds_at_mean_age": float(slope),
            "se_slope": se_slope,
            "z": z,
            "p": p,
            "odds_ratio_per_unit_age": or_ratio,
            "odds_ratio_95ci": [ci_low, ci_high]
        }

    # Add reference site (the omitted category)
    # Reference slope is beta_age alone
    try:
        var_ref = var_age
        se_ref = float(np.sqrt(var_ref)) if (var_ref is not None and not np.isnan(var_ref)) else float('nan')
    except Exception:
        se_ref = float('nan')
    z_ref = beta_age / se_ref if (se_ref and not np.isnan(se_ref) and se_ref > 0) else float('nan')
    p_ref = float(z_to_p(z_ref)) if not np.isnan(z_ref) else float('nan')
    or_ref = float(np.exp(beta_age))
    ci_ref_low = float(np.exp(beta_age - 1.96 * se_ref)) if (not np.isnan(se_ref)) else None
    ci_ref_high = float(np.exp(beta_age + 1.96 * se_ref)) if (not np.isnan(se_ref)) else None

    reference_info = {
        "age_param": age_param,
        "slope_log_odds_at_mean_age": float(beta_age),
        "se_slope": se_ref,
        "z": z_ref,
        "p": p_ref,
        "odds_ratio_per_unit_age": or_ref,
        "odds_ratio_95ci": [ci_ref_low, ci_ref_high],
        "note": "This corresponds to the model reference (omitted) site. Other sites are differences from this reference via interaction terms."
    }

    # Age_sq info
    if age_sq_param is not None and age_sq_param in params.index:
        beta_age_sq = float(params[age_sq_param])
        try:
            var_age_sq = cov_get(age_sq_param, age_sq_param)
            se_age_sq = float(np.sqrt(var_age_sq))
        except Exception:
            se_age_sq = float('nan')
        z_age_sq = beta_age_sq / se_age_sq if (se_age_sq and not np.isnan(se_age_sq) and se_age_sq > 0) else float('nan')
        p_age_sq = float(z_to_p(z_age_sq)) if not np.isnan(z_age_sq) else float('nan')
        # CI on coefficient scale
        try:
            ci_age_sq = [float(params[age_sq_param] - 1.96 * se_age_sq), float(params[age_sq_param] + 1.96 * se_age_sq)]
        except Exception:
            ci_age_sq = [None, None]
        age_sq_info = {
            "param_name": age_sq_param,
            "beta": beta_age_sq,
            "se": se_age_sq,
            "z": z_age_sq,
            "p": p_age_sq,
            "95ci": ci_age_sq,
            "interpretation": "A significant Age_sq indicates non-linearity in the age effect; effect of age depends on (mean-centered) age."
        }
    else:
        age_sq_info = {"param_name": age_sq_param, "note": "No Age_sq parameter found in the model."}

    notes = []
    if not sites:
        notes.append("No site-level parameters were parsed from the model parameter names. That may indicate SiteID had only one level or parameter naming differs.")
    else:
        notes.append(f"Parsed {len(sites)} site-specific interaction(s). The reference site (omitted) slope is provided under 'reference_site'.")
    notes.append("Slopes are reported at Age_c = 0 (mean-centered age). The quadratic Age_sq coefficient (if present) affects slopes at ages != mean.")

    out_object = {
        "age_sq": age_sq_info,
        "reference_site": reference_info,
        "sites": results_per_site,
        "notes": notes,
        "raw_params": {n: float(params[n]) for n in param_names}  # compact raw param snapshot
    }

    # Construct a concise description
    description_lines = []
    description_lines.append(
        "Returned per-site age slopes are the change in log-odds (and odds ratio) of choosing the majority option "
        "for a one-unit increase in mean-centered age (Age_c) evaluated at the mean age (Age_c = 0)."
    )
    description_lines.append(
        "The 'reference_site' entry is the age slope for the omitted/reference site (this equals the Age_c main effect). "
        "Each listed site adds its interaction coefficient to the reference slope to get the site-specific slope."
    )
    if age_sq_param is not None:
        description_lines.append(
            "Because the model also includes Age_sq, the true instantaneous effect of age depends on age (slope = beta_Age_c + 2*beta_Age_sq*Age_c + interaction). "
            "The slopes returned here are the special case at Age_c = 0 (mean age), where the quadratic term contributes zero."
        )
    else:
        description_lines.append("No quadratic age effect was found in the model; reported slopes represent the constant linear age effect at mean age.")
    description_lines.append("P-values are approximate (normal/z-test) and confidence intervals for odds ratios are exp(beta +/- 1.96*SE).")

    return {
        "object": out_object,
        "description": " ".join(description_lines)
    }