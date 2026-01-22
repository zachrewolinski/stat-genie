def extract_final_answer(model_output):
    """
    Extracts age-related effects (coefficients, SE, z, p, OR, 95% CI) from two fitted GLM models:
      - model_social_use: predicting SocialUse ~ AgeYears * C(Site) + ...
      - model_majority_pref: predicting MajorityPreference ~ AgeYears * C(Site) + ...
    
    Returns a dict with:
      - "object": a dictionary summarizing per-site age slopes and statistics for each model
      - "description": a short explanation of what the numbers mean in context
    """
    import numpy as np
    import re
    from math import exp
    from scipy.stats import norm

    def safe_get_levels(res):
        # Try to get Site levels from the model's data.frame (most reliable)
        try:
            df = res.model.data.frame
            if 'Site' in df:
                col = df['Site']
                # If categorical, return categories in order; otherwise return unique values in appearance order
                if hasattr(col, 'cat'):
                    return list(col.cat.categories)
                else:
                    # preserve observed order
                    uniques = list(dict.fromkeys(list(col)))
                    return uniques
        except Exception:
            pass
        # Fallback: parse parameter names to get the non-reference levels (can't identify reference reliably)
        params = res.params.index.tolist()
        parsed = []
        for nm in params:
            m = re.search(r'C\(Site\)\[T\.([^\]]+)\]', nm)
            if m:
                parsed.append(m.group(1))
        # If parsed is non-empty, assume there is a reference level but we don't know its name.
        if parsed:
            # Create a placeholder for the reference level
            return ['<reference>'] + parsed
        # Last resort: return an empty list
        return []

    def summarize_age_by_site(res):
        """
        For a fitted results object `res`, compute per-site age slopes (log-odds increase per year)
        and related statistics. Returns dict keyed by site with entries:
          coef, se, z, p, OR, CI_lower, CI_upper
        """
        out = {}
        params = res.params
        cov = res.cov_params()
        # Baseline AgeYears coefficient must exist for the model that included AgeYears
        if 'AgeYears' not in params.index:
            raise ValueError("Model does not contain 'AgeYears' term in parameters.")
        beta_age = float(params['AgeYears'])
        var_age = float(cov.loc['AgeYears', 'AgeYears'])
        se_age = np.sqrt(var_age)
        # find site levels
        levels = safe_get_levels(res)
        # Determine the actual reference site name if possible:
        # If levels came from data.frame and length>0, the first is used as reference by patsy/statsmodels
        # If we only parsed param names, levels[0] is '<reference>'
        if len(levels) == 0:
            # Try to infer by counting C(Site) parameters: get names and derive levels seen
            # fallback: handle only baseline
            levels = ['<reference>']
        reference = levels[0]
        # For each level compute slope
        for site in levels:
            if site == reference:
                coef = beta_age
                se = se_age
            else:
                # look for interaction param naming variations
                interaction_names = [
                    f'AgeYears:C(Site)[T.{site}]',
                    f'C(Site)[T.{site}]:AgeYears'
                ]
                inter_name = None
                for nm in interaction_names:
                    if nm in params.index:
                        inter_name = nm
                        break
                if inter_name is not None:
                    beta_inter = float(params[inter_name])
                    # compute var(beta_age + beta_inter) = var_age + var_inter + 2*cov
                    var_inter = float(cov.loc[inter_name, inter_name])
                    cov_ai = float(cov.loc['AgeYears', inter_name])
                    coef = beta_age + beta_inter
                    var_sum = var_age + var_inter + 2.0 * cov_ai
                    # numerical safety: var_sum >= 0
                    var_sum = max(var_sum, 0.0)
                    se = np.sqrt(var_sum)
                else:
                    # interaction missing -> slope equals baseline slope
                    coef = beta_age
                    se = se_age
            z = coef / se if se > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if se > 0 else np.nan
            OR = exp(coef)
            ci_low = exp(coef - 1.96 * se) if se > 0 else np.nan
            ci_high = exp(coef + 1.96 * se) if se > 0 else np.nan
            out[site] = {
                'coef_log_odds_per_year': coef,
                'se': se,
                'z': z,
                'p_value': p,
                'odds_ratio_per_year': OR,
                'OR_95CI_lower': ci_low,
                'OR_95CI_upper': ci_high
            }
        # Also include baseline term info for clarity
        out['_baseline_term'] = {
            'reference_site': reference,
            'AgeYears_coef': beta_age,
            'AgeYears_se': se_age,
            'AgeYears_z': beta_age / se_age if se_age > 0 else np.nan,
            'AgeYears_p': 2 * (1 - norm.cdf(abs(beta_age / se_age))) if se_age > 0 else np.nan
        }
        return out

    # Extract models from input
    res1 = model_output.get('model_social_use')
    res2 = model_output.get('model_majority_pref')

    result_obj = {}
    try:
        result_obj['model_social_use'] = summarize_age_by_site(res1)
    except Exception as e:
        result_obj['model_social_use'] = {'error': str(e)}

    try:
        result_obj['model_majority_pref'] = summarize_age_by_site(res2)
    except Exception as e:
        result_obj['model_majority_pref'] = {'error': str(e)}

    # Add sample sizes if provided
    if 'df_used_for_model_1_shape' in model_output:
        try:
            result_obj['n_model_1'] = int(model_output['df_used_for_model_1_shape'][0])
        except Exception:
            pass
    if 'df_used_for_model_2_shape' in model_output:
        try:
            result_obj['n_model_2'] = int(model_output['df_used_for_model_2_shape'][0])
        except Exception:
            pass

    # Short description interpreting the returned numbers
    description = (
        "Returned per-site estimates of the age slope (log-odds change per additional year) "
        "for (1) reliance on social information (SocialUse) and (2) majority preference among social learners "
        "(MajorityPreference). For each site you get: coef_log_odds_per_year, se, z, p_value, "
        "odds_ratio_per_year, and a 95% CI for the odds ratio. The '_baseline_term' entry shows which "
        "site is the reference (the category omitted from the C(Site) dummy coding) and the baseline AgeYears "
        "coefficient. Small p-values (e.g., < .05) indicate a statistically significant developmental trend "
        "in that site's slope (or in the baseline slope if the interaction term is absent)."
    )

    return {"object": result_obj, "description": description}