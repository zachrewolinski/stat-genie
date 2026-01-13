def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of age (linear and quadratic) and Age x Site interactions
    from a fitted statsmodels GLM (logistic) model.

    Returns a dictionary with keys:
      - "object": a dict containing numeric summaries (coefficients, SE, z, p, 95% CI, per-site slopes, ORs)
      - "description": a brief plain-language interpretation of what these numbers mean w.r.t.
                       how reliance on majority preference changes with age across sites.

    Assumptions:
      - model_output is a statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
      - The model formula included 'Age_c' and 'Age_c2' and Age_c by Site interactions (Age_c:C(Site)...)
    """
    import re
    import numpy as np
    from scipy.stats import norm

    res = model_output
    params = res.params
    cov = res.cov_params()

    # Safe exponential that avoids raising OverflowError and returns None for invalid inputs
    def safe_exp(x):
        try:
            if x is None:
                return None
            # Convert to numpy scalar if possible
            x_arr = np.asarray(x, dtype=float)
            if np.isnan(x_arr):
                return None
            val = np.exp(x_arr)
            # numpy returns inf for large positive inputs, 0.0 for large negative inputs
            if np.isfinite(val):
                return float(val)
            if val == np.inf:
                return float('inf')
            if val == -np.inf:
                return 0.0
            # fallback
            return float(val)
        except OverflowError:
            return float('inf')
        except Exception:
            return None

    # Helper to safely get param values (returns None if missing)
    def get_param(name):
        try:
            return params[name] if name in params.index else None
        except Exception:
            return None

    def get_cov(a, b):
        # return covariance between param a and b (0 if not present)
        try:
            if a in cov.index and b in cov.columns:
                return cov.loc[a, b]
            else:
                return 0.0
        except Exception:
            return 0.0

    # Basic age coefficients
    age_name = 'Age_c'
    age2_name = 'Age_c2'
    age_coef = float(get_param(age_name)) if get_param(age_name) is not None else None
    age2_coef = float(get_param(age2_name)) if get_param(age2_name) is not None else None

    # SE, z, p, CI for age linear
    results_obj = {}

    if age_coef is not None:
        var_age = float(get_cov(age_name, age_name))
        se_age = float(np.sqrt(var_age)) if var_age >= 0 else np.nan
        z_age = age_coef / se_age if se_age and se_age > 0 else np.nan
        p_age = 2 * (1 - norm.cdf(abs(z_age))) if not np.isnan(z_age) else np.nan
        ci_low_age = age_coef - 1.96 * se_age if not np.isnan(se_age) else np.nan
        ci_high_age = age_coef + 1.96 * se_age if not np.isnan(se_age) else np.nan
        results_obj['age_linear'] = {
            'param_name': age_name,
            'coef_logodds_per_year': float(age_coef),
            'se': se_age,
            'z': z_age,
            'p_value': p_age,
            '95ci_logodds': [ci_low_age, ci_high_age],
            'odds_ratio_per_year': safe_exp(age_coef),
            '95ci_odds_ratio': [safe_exp(ci_low_age), safe_exp(ci_high_age)]
        }
    else:
        results_obj['age_linear'] = None

    # Age quadratic
    if age2_coef is not None:
        var_age2 = float(get_cov(age2_name, age2_name))
        se_age2 = float(np.sqrt(var_age2)) if var_age2 >= 0 else np.nan
        z_age2 = age2_coef / se_age2 if se_age2 and se_age2 > 0 else np.nan
        p_age2 = 2 * (1 - norm.cdf(abs(z_age2))) if not np.isnan(z_age2) else np.nan
        ci_low_age2 = age2_coef - 1.96 * se_age2 if not np.isnan(se_age2) else np.nan
        ci_high_age2 = age2_coef + 1.96 * se_age2 if not np.isnan(se_age2) else np.nan
        results_obj['age_quadratic'] = {
            'param_name': age2_name,
            'coef_logodds_per_year2': float(age2_coef),
            'se': se_age2,
            'z': z_age2,
            'p_value': p_age2,
            '95ci_logodds': [ci_low_age2, ci_high_age2]
        }
    else:
        results_obj['age_quadratic'] = None

    # Find interaction parameter names for Age_c x Site
    # Typical naming from patsy/statsmodels: 'Age_c:C(Site)[T.SiteName]' or 'Age_c:C(Site)[T.Site]'
    inter_names = [n for n in params.index if ('Age_c' in n) and ('C(Site)' in n)]
    # also accept 'Age_c:C(Site)[T.' or 'Age_c: C(Site)[T.' spacing variants
    if not inter_names:
        inter_names = [n for n in params.index if re.search(r'Age_c.*C\(Site\)', n)]

    # Try to get site categories and reference level (if original data available)
    ref_site = None
    site_list = None
    try:
        df = res.model.data.frame
        if 'Site' in df.columns and hasattr(df['Site'].dtype, 'categories'):
            site_list = list(df['Site'].cat.categories)
            if len(site_list) > 0:
                ref_site = site_list[0]  # patsy default treatment coding uses first category as reference
    except Exception:
        # If we can't access original df, infer sites from param names
        pass

    # If we couldn't get site_list from data, attempt to parse from parameter names (C(Site) params)
    if site_list is None:
        site_param_names = [n for n in params.index if ('C(Site)' in n) and (':' not in n)]
        parsed_sites = []
        for n in site_param_names:
            m = re.search(r'C\(Site\)\[T\.?(.*)\]', n)
            if m:
                parsed_sites.append(m.group(1))
        if parsed_sites:
            # assume these are the non-reference sites; we cannot reliably infer reference name,
            # but we will create a list with reference = None followed by parsed sites
            site_list = [None] + parsed_sites
            ref_site = site_list[0]

    # Build per-site linear age slopes (log-odds per year)
    site_slopes = {}
    if age_coef is None:
        # can't compute slopes if Age_c missing
        site_slopes = None
    else:
        # reference site slope = Age_c
        if ref_site is not None:
            slope = age_coef
            var_slope = float(get_cov(age_name, age_name))
            se_slope = float(np.sqrt(var_slope)) if var_slope >= 0 else np.nan
            z_slope = slope / se_slope if se_slope and se_slope > 0 else np.nan
            p_slope = 2 * (1 - norm.cdf(abs(z_slope))) if not np.isnan(z_slope) else np.nan
            ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            site_slopes[ref_site] = {
                'slope_logodds_per_year': float(slope),
                'se': se_slope,
                'z': z_slope,
                'p_value': p_slope,
                '95ci_logodds': [ci_low, ci_high],
                'odds_ratio_per_year': safe_exp(slope),
                '95ci_odds_ratio': [safe_exp(ci_low), safe_exp(ci_high)]
            }

        # for each interaction term, compute slope = Age_c + interaction_coef
        for inter in inter_names:
            # parse site label
            m = re.search(r'Age_c:C\(Site\)\[T\.?(.*)\]', inter)
            site_label = m.group(1) if m else inter  # fallback to full name if we can't parse
            inter_val = get_param(inter)
            inter_coef = float(inter_val) if inter_val is not None else 0.0
            slope = age_coef + inter_coef
            # variance using var(age) + var(inter) + 2cov(age, inter)
            var_inter = float(get_cov(inter, inter))
            cov_ai = float(get_cov(age_name, inter))
            var_slope = float(get_cov(age_name, age_name)) + var_inter + 2.0 * cov_ai
            se_slope = float(np.sqrt(var_slope)) if var_slope >= 0 else np.nan
            z_slope = slope / se_slope if se_slope and se_slope > 0 else np.nan
            p_slope = 2 * (1 - norm.cdf(abs(z_slope))) if not np.isnan(z_slope) else np.nan
            ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan

            site_slopes[site_label] = {
                'interaction_param': inter,
                'interaction_coef': inter_coef,
                'slope_logodds_per_year': float(slope),
                'se': se_slope,
                'z': z_slope,
                'p_value': p_slope,
                '95ci_logodds': [ci_low, ci_high],
                'odds_ratio_per_year': safe_exp(slope),
                '95ci_odds_ratio': [safe_exp(ci_low), safe_exp(ci_high)]
            }

    results_obj['site_slopes'] = site_slopes

    # Wald test for joint significance of all Age_c:C(Site) interaction terms (are their coefficients all zero?)
    interaction_wald = None
    if inter_names:
        # build R matrix selecting those parameters
        k = len(params.index)
        idxs = [params.index.get_loc(n) for n in inter_names]
        R = np.zeros((len(idxs), k))
        for i, j in enumerate(idxs):
            R[i, j] = 1.0
        # Use statsmodels wald_test; this may return different result object types across versions
        try:
            wtest = res.wald_test(R)
            # extract attributes safely
            chi2_stat = None
            pval = None
            df_denom = None
            df_num = None
            try:
                # wtest may have statistic as array or scalar
                stat = getattr(wtest, 'statistic', None)
                if stat is not None:
                    # if it's an array-like, take scalar value
                    if hasattr(stat, 'item'):
                        chi2_stat = float(stat.item())
                    else:
                        chi2_stat = float(stat)
            except Exception:
                chi2_stat = None
            try:
                pval = float(getattr(wtest, 'pvalue', None)) if getattr(wtest, 'pvalue', None) is not None else None
            except Exception:
                pval = None
            try:
                df_num = int(getattr(wtest, 'df_num', None)) if getattr(wtest, 'df_num', None) is not None else None
            except Exception:
                df_num = None
            try:
                df_denom = int(getattr(wtest, 'df_denom', None)) if getattr(wtest, 'df_denom', None) is not None else None
            except Exception:
                df_denom = None

            interaction_wald = {
                'chi2_stat': chi2_stat,
                'df_denom': df_denom,
                'df_num': df_num,
                'p_value': pval,
                'description': 'Joint test: all Age_c x Site interaction coefficients = 0'
            }
        except Exception:
            interaction_wald = {
                'chi2_stat': None,
                'df_denom': None,
                'df_num': None,
                'p_value': None,
                'description': 'Wald test could not be computed for Age_c x Site interactions'
            }
    results_obj['interaction_wald_test'] = interaction_wald

    # Also include the raw coefficients and their p-values for Age_c and Age_c2 (for transparency)
    # Build a small coeff table for relevant params
    coeff_table = {}
    relevant_params = [p for p in params.index if p in (age_name, age2_name) or p in inter_names]
    for p in relevant_params:
        try:
            coef = float(params[p])
        except Exception:
            coef = None
        varp = float(get_cov(p, p))
        sep = float(np.sqrt(varp)) if varp >= 0 else np.nan
        z = coef / sep if (coef is not None and sep and sep > 0) else np.nan
        pval = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = coef - 1.96 * sep if (coef is not None and not np.isnan(sep)) else np.nan
        ci_high = coef + 1.96 * sep if (coef is not None and not np.isnan(sep)) else np.nan
        coeff_table[p] = {
            'coef': coef,
            'se': sep,
            'z': z,
            'p_value': pval,
            '95ci': [ci_low, ci_high]
        }

    results_obj['coeff_table_relevant'] = coeff_table

    # Prepare final return object and short description
    description_lines = []
    description_lines.append(
        "Extracted linear age effect (Age_c), quadratic age effect (Age_c2), and per-site linear age slopes "
        "which combine the Age_c main effect and Age_c x Site interaction coefficients."
    )
    description_lines.append(
        "For each site, 'slope_logodds_per_year' is the estimated change in log-odds of choosing the majority option "
        "per additional year of age. 'odds_ratio_per_year' is exp(slope) (multiplicative change in odds per year)."
    )
    if interaction_wald and interaction_wald.get('p_value') is not None:
        try:
            pv = interaction_wald['p_value']
            if pv < 0.05:
                description_lines.append(
                    f"The joint Wald test for Age_c x Site interactions is significant (p = {pv:.3g}), "
                    "indicating developmental slopes differ across sites."
                )
            else:
                description_lines.append(
                    f"The joint Wald test for Age_c x Site interactions is not significant (p = {pv:.3g}), "
                    "indicating no strong evidence that developmental slopes differ across sites."
                )
        except Exception:
            description_lines.append("The joint Wald test result was retrieved but could not be interpreted numerically.")
    else:
        description_lines.append("No Age_c x Site interaction terms were found in the model or the joint test could not be computed.")

    description = " ".join(description_lines)

    return {
        "object": results_obj,
        "description": description
    }