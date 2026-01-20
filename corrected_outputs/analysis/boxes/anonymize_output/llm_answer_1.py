def extract_final_answer(model_output):
    """
    Extract per-site age slopes (effect of Age_c on probability of choosing majority)
    from a fitted statsmodels logistic model with Age_c * C(Site) interactions.

    Returns a dict:
      - "object": dict mapping each Site -> statistics (slope log-odds, SE, z, p, 95% CI,
                    odds ratio and OR 95% CI)
      - "description": brief interpretation of what the returned numbers mean.

    Notes:
      - Requires model_output to be a statsmodels results object (BinaryResultsWrapper
        or a robustcov results wrapper) with .params and .cov_params() available.
      - Assumes the model used a formula with 'Age_c' and an interaction like
        'Age_c:C(Site)[T.<level>]'. The reference site (omitted level) will be included
        with its slope equal to the Age_c main effect.
    """
    import numpy as np
    import pandas as pd
    from math import exp
    try:
        from scipy import stats
        norm_sf = lambda x: stats.norm.sf(x)
    except Exception:
        # fallback using approximate normal cdf via math.erf if scipy not available
        import math
        def norm_sf(x):
            return 0.5 - 0.5 * math.erf(x / math.sqrt(2))

    out = {}
    try:
        params = model_output.params  # pandas Series
        cov = model_output.cov_params()  # DataFrame
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read params/covariance from model_output: {e}"
        }

    # Get list of site levels from the original data if possible
    site_levels = None
    try:
        df = model_output.model.data.frame
        if 'Site' in df.columns:
            site_levels = list(pd.Categorical(df['Site']).categories)
    except Exception:
        site_levels = None

    # Base Age coefficient name
    age_name = 'Age_c'
    if age_name not in params.index:
        # try alternative naming patterns
        candidate = [n for n in params.index if n.strip() == 'Age_c']
        if candidate:
            age_name = candidate[0]
        else:
            return {
                "object": None,
                "description": "Model does not contain a coefficient named 'Age_c'; cannot extract age effect."
            }

    age_coef = float(params[age_name])
    var_age = float(cov.loc[age_name, age_name])

    # Find interaction terms linking Age_c and Site
    # Possible patterns: 'Age_c:C(Site)[T.xxx]' or 'Age_c:C(Site)[T.xxx]' or 'Age_c:C(Site)[T.xxx]'
    interaction_map = {}  # site_level -> param_name
    for pname in params.index:
        if 'Age_c' in pname and 'C(Site)' in pname:
            # extract level name between 'C(Site)[T.' and ']'
            p = pname
            if 'C(Site)[T.' in p:
                try:
                    lvl = p.split('C(Site)[T.')[1].split(']')[0]
                except Exception:
                    lvl = p
            else:
                # fallback: take full pname as key
                lvl = p
            interaction_map[lvl] = pname

    # If site_levels not available from data, attempt to assemble from parameter names:
    if site_levels is None:
        # include levels that appear in interaction_map keys, and add a placeholder for reference level
        site_levels = list(interaction_map.keys())
        # try to find reference level by looking at 'C(Site)[T.<level>]' categories used in other params
        # but if we can't determine reference, label it "reference".
        # We will include a 'reference' site representing the omitted baseline level.
        site_levels = ['(reference)'] + site_levels

    results = {}
    for site in site_levels:
        if site == '(reference)':
            # reference site's slope = age_coef
            slope = age_coef
            se = np.sqrt(var_age)
            # compute CI and p
            z = slope / se if se > 0 else np.nan
            p = 2 * norm_sf(abs(z)) if se > 0 else np.nan
        else:
            if site in interaction_map:
                inter_name = interaction_map[site]
                inter_coef = float(params[inter_name])
                slope = age_coef + inter_coef
                # variance: Var(A)+Var(I)+2 Cov(A,I)
                try:
                    cov_ai = float(cov.loc[age_name, inter_name])
                    var_inter = float(cov.loc[inter_name, inter_name])
                    var_slope = var_age + var_inter + 2.0 * cov_ai
                    se = np.sqrt(var_slope) if var_slope >= 0 else np.nan
                except Exception:
                    # if cov not available, fall back to NA
                    se = np.nan
                z = slope / se if (se is not None and se and not np.isnan(se)) else np.nan
                p = 2 * norm_sf(abs(z)) if (not np.isnan(z)) else np.nan
            else:
                # No interaction term found for this site — assume same slope as reference
                slope = age_coef
                se = np.sqrt(var_age)
                z = slope / se if se > 0 else np.nan
                p = 2 * norm_sf(abs(z)) if se > 0 else np.nan

        # 95% CI on log-odds scale
        if se is not None and not np.isnan(se):
            ci_low = slope - 1.96 * se
            ci_high = slope + 1.96 * se
            # convert to odds ratio scale
            try:
                or_point = float(exp(slope))
                or_low = float(exp(ci_low))
                or_high = float(exp(ci_high))
            except Exception:
                or_point = or_low = or_high = np.nan
        else:
            ci_low = ci_high = or_point = or_low = or_high = np.nan

        results[str(site)] = {
            "slope_log_odds_per_year": float(slope) if not np.isnan(slope) else None,
            "se_slope": float(se) if not np.isnan(se) else None,
            "z": float(z) if not np.isnan(z) else None,
            "p_two_tailed": float(p) if not np.isnan(p) else None,
            "95%CI_log_odds": [float(ci_low) if not np.isnan(ci_low) else None,
                               float(ci_high) if not np.isnan(ci_high) else None],
            "odds_ratio_per_year": float(or_point) if not np.isnan(or_point) else None,
            "95%CI_odds_ratio": [float(or_low) if not np.isnan(or_low) else None,
                                 float(or_high) if not np.isnan(or_high) else None]
        }

    description_lines = [
        "For each Site, the returned 'slope_log_odds_per_year' is the estimated change in log-odds",
        "of choosing the majority option associated with a one-year increase in age in that site.",
        "Positive slopes indicate increasing reliance on the majority with age (OR > 1);",
        "negative slopes indicate decreasing reliance (OR < 1).",
        "Columns include standard error, z-statistic, two-tailed p-value, and 95% CIs both on the",
        "log-odds scale and transformed to odds ratios. The '(reference)' site corresponds to",
        "the baseline category of Site used by the model (its Age_c coefficient is the model's Age_c main effect)."
    ]
    description = " ".join(description_lines)

    return {"object": results, "description": description}