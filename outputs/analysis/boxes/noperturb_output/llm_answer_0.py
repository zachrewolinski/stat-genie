def extract_final_answer(model_output):
    """
    Extract age-related effects (main + culture interactions) from the fitted models.
    Inputs:
      - model_output: dict with keys 'multinomial', 'social_follow_logit', 'majority_pref_logit'
                      each holding a statsmodels fitted results object (or None).
    Returns:
      - dict with keys:
          "object": nested dict of extracted numeric results (coefficients, SEs, p-values,
                    95% CIs, and culture-specific combined age slopes where computable)
          "description": textual explanation of what the returned numbers mean.
    Notes:
      - The function expects the original model to have used variable names:
          'age_c' for centered continuous age,
          culture dummy columns named like 'culture_2', 'culture_3', ...
          and interaction columns named like 'age_c:culture_2', etc.
      - For the multinomial model we return the age coefficients per outcome (and
        interaction coefficients). Where possible we also give the combined age slope
        for each culture by adding the main age coefficient and the interaction coef
        for that culture (and computing SE via the model covariance). If that cannot
        be computed from the covariance structure, the separate terms are still returned.
    """
    import numpy as np
    import pandas as pd
    from math import sqrt
    from scipy import stats

    results_out = {}

    def safe_conf_int_from_res(res, param_name):
        try:
            ci = res.conf_int().loc[param_name].tolist()
            return float(ci[0]), float(ci[1])
        except Exception:
            return None

    def get_culture_dummies_from_params(param_index):
        # param_index: iterable of parameter names (strings)
        # return list of culture dummy variable names like 'culture_2', ...
        return sorted([p for p in param_index if isinstance(p, str) and p.startswith('culture_')])

    def compute_combined_and_stats(res, age_var, interaction_var):
        """
        For a BinaryResultsWrapper (Logit), compute combined effect = coef(age_var) + coef(interaction_var)
        using covariance matrix to get se, z, p, ci. If interaction_var not present, it's taken as 0.
        Returns dict with separate coef/se/p/ci for main and interaction and combined values (if possible).
        """
        out = {}
        params = res.params
        pvals = res.pvalues
        bse = res.bse
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        # main age
        if age_var in params.index:
            coef_age = float(params[age_var])
            se_age = float(bse[age_var]) if age_var in bse.index else None
            p_age = float(pvals[age_var]) if age_var in pvals.index else None
            ci_age = safe_conf_int_from_res(res, age_var)
        else:
            coef_age = se_age = p_age = ci_age = None

        # interaction
        if interaction_var in params.index:
            coef_int = float(params[interaction_var])
            se_int = float(bse[interaction_var]) if interaction_var in bse.index else None
            p_int = float(pvals[interaction_var]) if interaction_var in pvals.index else None
            ci_int = safe_conf_int_from_res(res, interaction_var)
        else:
            coef_int = 0.0
            se_int = 0.0
            p_int = None
            ci_int = None

        out['age_main'] = {'coef': coef_age, 'se': se_age, 'p': p_age, 'ci95': ci_age}
        out['age_interaction'] = {'name': interaction_var, 'coef': coef_int, 'se': se_int, 'p': p_int, 'ci95': ci_int}

        # combined
        combined = None
        combined_se = None
        combined_p = None
        combined_ci = None
        if (coef_age is not None) and (cov is not None):
            try:
                # compute var(sum) = var(age) + var(int) + 2cov(age,int)
                # cov is DataFrame indexed by param names
                # If interaction_var absent, treat coef_int=0 and var_int=0 and cov=0
                var_age = float(cov.loc[age_var, age_var]) if (age_var in cov.index and age_var in cov.columns) else 0.0
                if interaction_var in cov.index and interaction_var in cov.columns:
                    var_int = float(cov.loc[interaction_var, interaction_var])
                    cov_age_int = float(cov.loc[age_var, interaction_var]) if (age_var in cov.index and interaction_var in cov.columns) else 0.0
                else:
                    var_int = 0.0
                    cov_age_int = 0.0
                combined = coef_age + coef_int
                combined_var = var_age + var_int + 2.0 * cov_age_int
                combined_se = sqrt(max(combined_var, 0.0))
                z = combined / combined_se if combined_se > 0 else None
                combined_p = float(2 * (1 - stats.norm.cdf(abs(z)))) if z is not None else None
                combined_ci = (combined - 1.96 * combined_se, combined + 1.96 * combined_se) if combined_se is not None else None
            except Exception:
                combined = coef_age + coef_int
                combined_se = None
                combined_p = None
                combined_ci = None
        else:
            # cannot compute se from cov, but still give point estimate
            if coef_age is not None:
                combined = coef_age + coef_int
        out['combined'] = {'coef': combined, 'se': combined_se, 'p': combined_p, 'ci95': combined_ci}
        return out

    # 1) Multinomial model: extract age-related params and interactions per outcome
    mn = model_output.get('multinomial', None)
    multinomial_info = {}
    if mn is None:
        multinomial_info['note'] = 'No multinomial model provided.'
    else:
        try:
            # params: DataFrame with rows = outcome levels (excluded base is not included), cols = exog names
            params_df = mn.params.copy()
            pvals_df = mn.pvalues.copy()
            # Determine outcome labels (index of params_df)
            outcomes = list(params_df.index)
            multinomial_info['outcomes'] = {}
            # culture dummy names found in columns
            culture_cols = get_culture_dummies_from_params(params_df.columns)
            multinomial_info['culture_dummies'] = culture_cols
            multinomial_info['note'] = ("For each non-baseline outcome we return the main age coefficient "
                                        "and any age:culture interaction coefficients. The combined culture-specific "
                                        "age slope equals (age main) + (age:culture_k) and must be computed by summation; "
                                        "where possible a combined SE/p-value is also computed below.")
            cov = None
            try:
                cov = mn.cov_params()
            except Exception:
                cov = None
            # For each outcome (e.g., majority, minority rows), extract age main & interactions
            for outcome in outcomes:
                row = params_df.loc[outcome]
                row_p = pvals_df.loc[outcome] if (hasattr(pvals_df, 'loc') and outcome in pvals_df.index) else None
                outcome_dict = {}
                # main age coefficient
                if 'age_c' in row.index:
                    outcome_dict['age_main'] = {'coef': float(row['age_c']),
                                                'p': (float(row_p['age_c']) if row_p is not None and 'age_c' in row_p.index else None)}
                else:
                    outcome_dict['age_main'] = {'coef': None, 'p': None}
                # interactions
                interactions = {}
                for c in culture_cols:
                    inter_name = f'age_c:{c}'
                    if inter_name in row.index:
                        interactions[c] = {'interaction_name': inter_name,
                                           'coef': float(row[inter_name]),
                                           'p': (float(row_p[inter_name]) if row_p is not None and inter_name in row_p.index else None)}
                    else:
                        interactions[c] = {'interaction_name': inter_name, 'coef': 0.0, 'p': None}
                outcome_dict['age_interactions'] = interactions

                # Try to compute combined age slope for base (reference) and for each culture
                combined_slopes = {}
                # Determine the "reference" culture label: we cannot infer exact id of omitted culture from params alone,
                # so we call it 'reference' (the culture that has no dummy column).
                combined_slopes['reference'] = None
                # Combined for reference is simply the age_main
                try:
                    main_coef = float(row['age_c']) if 'age_c' in row.index else None
                except Exception:
                    main_coef = None
                if main_coef is not None:
                    combined_slopes['reference'] = {'coef': main_coef, 'p': (float(row_p['age_c']) if row_p is not None and 'age_c' in row_p.index else None)}
                # For each culture dummy, combined = age_main + that interaction
                for c in culture_cols:
                    inter_name = f'age_c:{c}'
                    inter_coef = float(row[inter_name]) if inter_name in row.index else 0.0
                    combined_point = (main_coef + inter_coef) if main_coef is not None else None
                    combined_entry = {'coef': combined_point}
                    # Try to compute SE and p using cov (cov has multi-index or flattened names). If not possible, leave None.
                    if cov is not None and main_coef is not None:
                        try:
                            # Attempt to locate covariance entries. cov may have MultiIndex (outcome, param) or flat names.
                            # Build keys matching how mn.params are laid out. We'll try a few heuristics.
                            # Preferred key form if cov is MultiIndex: (outcome, 'age_c')
                            if isinstance(cov.index, pd.MultiIndex):
                                key_age = (outcome, 'age_c')
                                key_int = (outcome, inter_name) if (outcome, inter_name) in cov.index else (outcome, inter_name)
                                var_age = float(cov.loc[key_age, key_age])
                                var_int = float(cov.loc[key_int, key_int]) if key_int in cov.index else 0.0
                                cov_ai = float(cov.loc[key_age, key_int]) if key_int in cov.index else 0.0
                            else:
                                # cov.index is flat. Find indices that contain both outcome and param name.
                                # Common flattened formats include "1.age_c" or "age_c[1]" etc. We'll search substrings.
                                def find_idx_for(outcome_label, param_label):
                                    matches = [idx for idx in cov.index.astype(str) if (str(outcome_label) in str(idx)) and (param_label in str(idx))]
                                    if len(matches) == 1:
                                        return matches[0]
                                    # fallback: find entries that end with param_label
                                    matches2 = [idx for idx in cov.index.astype(str) if str(idx).endswith(param_label)]
                                    return matches2[0] if matches2 else None
                                idx_age = find_idx_for(outcome, 'age_c')
                                idx_int = find_idx_for(outcome, inter_name)
                                if idx_age is None:
                                    # maybe param names are just flattened as 'age_c' (if only one eq). Then try 'age_c'
                                    idx_age = 'age_c' if 'age_c' in cov.index else None
                                if idx_int is None:
                                    idx_int = inter_name if inter_name in cov.index else None
                                if (idx_age is not None) and (idx_age in cov.index):
                                    var_age = float(cov.loc[idx_age, idx_age])
                                else:
                                    var_age = 0.0
                                if (idx_int is not None) and (idx_int in cov.index):
                                    var_int = float(cov.loc[idx_int, idx_int])
                                    cov_ai = float(cov.loc[idx_age, idx_int]) if (idx_age in cov.index and idx_int in cov.index) else 0.0
                                else:
                                    var_int = 0.0
                                    cov_ai = 0.0
                            combined_var = var_age + var_int + 2.0 * cov_ai
                            combined_se = sqrt(max(combined_var, 0.0))
                            combined_z = combined_point / combined_se if (combined_se is not None and combined_se > 0) else None
                            combined_p = float(2 * (1 - stats.norm.cdf(abs(combined_z)))) if combined_z is not None else None
                            combined_ci = (combined_point - 1.96 * combined_se, combined_point + 1.96 * combined_se) if combined_se is not None else None
                            combined_entry.update({'se': combined_se, 'p': combined_p, 'ci95': combined_ci})
                        except Exception:
                            # if anything fails, leave se/p as None
                            combined_entry.update({'se': None, 'p': None, 'ci95': None})
                    else:
                        combined_entry.update({'se': None, 'p': None, 'ci95': None})
                    combined_slopes[c] = combined_entry

                outcome_dict['combined_age_slopes'] = combined_slopes
                multinomial_info['outcomes'][str(outcome)] = outcome_dict

        except Exception as e:
            multinomial_info['error'] = f'Could not extract multinomial info: {repr(e)}'

    results_out['multinomial'] = multinomial_info

    # 2) social_follow logistic
    sf = model_output.get('social_follow_logit', None)
    sf_info = {}
    if sf is None:
        sf_info['note'] = 'No social_follow_logit model provided.'
    else:
        try:
            params = sf.params
            pvals = sf.pvalues
            bse = sf.bse
            cov = None
            try:
                cov = sf.cov_params()
            except Exception:
                cov = None
            # Identify culture dummies from params
            culture_cols = get_culture_dummies_from_params(params.index)
            sf_info['culture_dummies'] = culture_cols
            sf_info['age_main'] = {'coef': float(params['age_c']) if 'age_c' in params.index else None,
                                   'se': float(bse['age_c']) if ('age_c' in bse.index) else None,
                                   'p': float(pvals['age_c']) if ('age_c' in pvals.index) else None,
                                   'ci95': safe_conf_int_from_res(sf, 'age_c')}
            # For each culture dummy compute combined age slope
            combined_by_culture = {}
            for c in culture_cols:
                inter_name = f'age_c:{c}'
                combined_stats = compute_combined_and_stats(sf, 'age_c', inter_name)
                combined_by_culture[c] = combined_stats
            # Also include reference culture (no interaction)
            combined_by_culture['reference'] = {'age_main': sf_info['age_main'], 'combined': {'coef': float(params['age_c']) if 'age_c' in params.index else None}}
            sf_info['combined_age_by_culture'] = combined_by_culture
        except Exception as e:
            sf_info['error'] = f'Could not extract social_follow info: {repr(e)}'
    results_out['social_follow'] = sf_info

    # 3) majority_pref logistic (may be None)
    mp = model_output.get('majority_pref_logit', None)
    mp_info = {}
    if mp is None:
        mp_info['note'] = 'No majority_pref_logit model provided or not enough data to fit.'
    else:
        try:
            params = mp.params
            pvals = mp.pvalues
            bse = mp.bse
            cov = None
            try:
                cov = mp.cov_params()
            except Exception:
                cov = None
            culture_cols = get_culture_dummies_from_params(params.index)
            mp_info['culture_dummies'] = culture_cols
            mp_info['age_main'] = {'coef': float(params['age_c']) if 'age_c' in params.index else None,
                                   'se': float(bse['age_c']) if ('age_c' in bse.index) else None,
                                   'p': float(pvals['age_c']) if ('age_c' in pvals.index) else None,
                                   'ci95': safe_conf_int_from_res(mp, 'age_c')}
            combined_by_culture = {}
            for c in culture_cols:
                inter_name = f'age_c:{c}'
                combined_stats = compute_combined_and_stats(mp, 'age_c', inter_name)
                combined_by_culture[c] = combined_stats
            combined_by_culture['reference'] = {'age_main': mp_info['age_main'], 'combined': {'coef': float(params['age_c']) if 'age_c' in params.index else None}}
            mp_info['combined_age_by_culture'] = combined_by_culture
        except Exception as e:
            mp_info['error'] = f'Could not extract majority_pref info: {repr(e)}'
    results_out['majority_pref'] = mp_info

    # Summary description
    description = (
        "Returned object contains, for each fitted model, the main age coefficient (age_c) and, where present, "
        "the age:culture_k interaction coefficients. For the two binary logistic models (social_follow and majority_pref) "
        "the function attempts to compute a combined culture-specific age slope = (age main) + (age:culture_k) and its SE/p-value "
        "using the model covariance matrix. For the multinomial model we return per-outcome age main and interaction coefficients and "
        "attempt to compute combined slopes per outcome & culture when the covariance structure permits. \n\n"
        "Interpretation guidance: A positive combined age slope for the 'majority' outcome indicates that with increasing age, "
        "children are more likely to choose the majority option (relative to the baseline category used in the multinomial model). "
        "For the binary models, a positive combined slope for social_follow indicates that older children are more likely to follow demonstrators; "
        "a positive combined slope for majority_pref indicates older children who follow demonstrators are more likely to prefer the majority. "
        "Statistical significance (p < .05) in the combined slope indicates a reliable age-related trend in that culture. "
        "If combined SE/p-values could not be computed due to covariance-format differences, please compute the combined estimate as "
        "age_coef + interaction_coef (both provided) and request a re-run if you need automatic SE/p-values for those combinations."
    )

    return {"object": results_out, "description": description}