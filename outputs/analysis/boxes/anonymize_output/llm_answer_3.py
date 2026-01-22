def extract_final_answer(model_output):
    """
    Extract age-related developmental slopes (effect of Age_c) on choosing the majority option
    for each cultural site from a fitted statsmodels logistic regression (majority_model).
    
    Input:
      - model_output: dict containing at least key 'majority_model' with a fitted
                      statsmodels BinaryResultsWrapper (or an Exception).
    
    Returns a dict with:
      - "object": dict mapping each site to the estimated age slope (log-odds per year),
                  its SE, z, p, 95% CI (log-odds) and odds ratio with 95% CI.
      - "description": short interpretation of what the numbers mean.
    """
    import numpy as np
    import math

    out = {"object": None, "description": ""}

    maj = model_output.get('majority_model', None)
    if maj is None:
        out["description"] = "No 'majority_model' found in model_output."
        return out

    # If an exception was stored instead of a fitted model, return it
    if isinstance(maj, Exception):
        out["description"] = "majority_model contains an Exception: " + repr(maj)
        out["object"] = {"error": repr(maj)}
        return out

    # Check that it's a fitted statsmodels results object
    try:
        params = maj.params          # pandas Series
        cov = maj.cov_params()       # DataFrame
        pvalues = maj.pvalues
    except Exception as e:
        out["description"] = "Could not read parameters from majority_model: " + repr(e)
        out["object"] = {"error": repr(e)}
        return out

    # Ensure Age_c is present
    if 'Age_c' not in params.index:
        out["description"] = "Model does not contain 'Age_c' term."
        out["object"] = {"available_params": list(params.index)}
        return out

    # Identify interaction parameter names that correspond to Age_c by site
    # statsmodels names interactions often like "Age_c:C(Site)[T.site_name]" or "Age_c:C(Site)[T.<site>]"
    interaction_terms = [name for name in params.index if ('Age_c' in name) and (':' in name or 'C(Site)' in name) and name != 'Age_c']
    # Fallback: also accept names containing "Age_c:C(Site)" explicitly
    # Parse site names from the interaction term strings
    nonref_sites = []
    for it in interaction_terms:
        # Try patterns like "Age_c:C(Site)[T.site]" or "Age_c:C(Site)[T.site_name]"
        if 'T.' in it:
            try:
                # take substring after 'T.' up to ] or end
                start = it.index('T.') + 2
                # find closing bracket if present
                end_idx = it.find(']', start)
                if end_idx == -1:
                    site_name = it[start:]
                else:
                    site_name = it[start:end_idx]
                nonref_sites.append(site_name)
            except Exception:
                # fallback: use full interaction term
                nonref_sites.append(it)
        else:
            # fallback: use entire interaction name
            nonref_sites.append(it)

    # Try to get the full list of Site levels from the original data if available
    site_levels = None
    try:
        df = maj.model.data.frame
        if df is not None and 'Site' in df.columns:
            # get unique preserving order
            site_levels = list(pd.unique(df['Site']))
    except Exception:
        site_levels = None

    # If we have site_levels and nonref_sites, deduce reference site
    ref_site = None
    if site_levels is not None and len(nonref_sites) >= 0:
        # convert to strings
        nonref_simple = [str(x) for x in nonref_sites]
        # find a site in site_levels not listed in nonref_sites
        ref_candidates = [str(s) for s in site_levels if str(s) not in nonref_simple]
        if len(ref_candidates) == 1:
            ref_site = ref_candidates[0]
        elif len(ref_candidates) > 1:
            # multiple candidates: choose first as reference
            ref_site = ref_candidates[0]
        else:
            # no candidates found -> cannot determine, mark as 'Reference'
            ref_site = 'Reference'
    else:
        # If site_levels not available, set generic label
        ref_site = 'Reference'

    # Prepare results dictionary
    results_by_site = {}

    # Base slope (reference site's Age_c effect)
    base_beta = float(params['Age_c'])
    base_var = float(cov.loc['Age_c', 'Age_c'])
    base_se = math.sqrt(base_var)
    base_z = base_beta / base_se if base_se > 0 else float('nan')
    # Prefer p-value from model if available for the Age_c alone term
    base_p = float(pvalues.get('Age_c', np.nan))
    base_ci_low = base_beta - 1.96 * base_se
    base_ci_high = base_beta + 1.96 * base_se
    base_or = math.exp(base_beta)
    base_or_ci = (math.exp(base_ci_low), math.exp(base_ci_high))

    results_by_site[ref_site] = {
        "log_odds_per_year": base_beta,
        "se": base_se,
        "z": base_z,
        "p": base_p,
        "95%_CI_log_odds": (base_ci_low, base_ci_high),
        "odds_ratio_per_year": base_or,
        "95%_CI_OR": base_or_ci,
        "note": "Reference site (intercept/site omitted in dummy coding)."
    }

    # For each non-reference site, compute combined slope = Age_c + interaction term
    for idx, site in enumerate(nonref_sites):
        # corresponding interaction param name is the interaction_terms[idx]
        # But safer: find matching interaction term name in params index that contains the site string
        match_name = None
        for name in params.index:
            if ('Age_c' in name) and (str(site) in name) and (name != 'Age_c'):
                match_name = name
                break
        if match_name is None:
            # fallback: use the interaction_terms list positionally
            if idx < len(interaction_terms):
                match_name = interaction_terms[idx]
            else:
                # cannot find matching interaction parameter
                results_by_site[str(site)] = {"error": f"Could not find interaction term for site {site}."}
                continue

        beta_inter = float(params[match_name])
        # variance of sum = var(Age_c) + var(inter) + 2*cov(Age_c, inter)
        try:
            var_inter = float(cov.loc[match_name, match_name])
            cov_ai = float(cov.loc['Age_c', match_name])
        except Exception:
            # If cov elements not found, fall back to NaN
            var_inter = np.nan
            cov_ai = np.nan

        combined_beta = base_beta + beta_inter
        if not (np.isnan(var_inter) or np.isnan(cov_ai)):
            combined_var = base_var + var_inter + 2 * cov_ai
            combined_se = math.sqrt(combined_var) if combined_var >= 0 else float('nan')
        else:
            combined_se = float('nan')

        combined_z = combined_beta / combined_se if (combined_se and not math.isnan(combined_se) and combined_se > 0) else float('nan')
        # compute p using normal approx if se available
        combined_p = float(2 * (1 - 0.5 * (1 + math.erf(abs(combined_z) / math.sqrt(2))))) if (not math.isnan(combined_z)) else float('nan')
        if not math.isnan(combined_se):
            ci_low = combined_beta - 1.96 * combined_se
            ci_high = combined_beta + 1.96 * combined_se
            or_val = math.exp(combined_beta)
            or_ci = (math.exp(ci_low), math.exp(ci_high))
        else:
            ci_low = ci_high = or_val = or_ci = (float('nan'), float('nan'))

        results_by_site[str(site)] = {
            "interaction_param": match_name,
            "log_odds_per_year": combined_beta,
            "se": combined_se,
            "z": combined_z,
            "p": combined_p,
            "95%_CI_log_odds": (ci_low, ci_high),
            "odds_ratio_per_year": or_val,
            "95%_CI_OR": or_ci
        }

    out["object"] = results_by_site

    # Short description
    out["description"] = (
        "For each site, the returned 'log_odds_per_year' is the estimated change in log-odds of choosing "
        "the majority option for a one-year increase in age. 'odds_ratio_per_year' is the exponentiated "
        "value (multiplicative change in odds per year). The reference site is the category omitted by "
        "dummy coding (labeled above). p-values and 95% CIs use normal approximation from the model covariance; "
        "if any covariance elements were unavailable the corresponding SE/CIs are NaN."
    )

    return out

# Note: This function expects the statsmodels results object to include .params, .cov_params(), and .pvalues.
# If you want to also return raw parameter table (coef, se, p) for Age_c and each Age_c:Site interaction,
# you can inspect maj.params and maj.pvalues directly.