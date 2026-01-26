def extract_final_answer(model_output):
    """
    Extracts statistics related to Age_c effects (main effect and Age_c * culture interactions)
    from the model_output produced by the modeling function.

    Parameters
    ----------
    model_output : dict
        Expected keys (some may be None): 'social_follow_model', 'majority_choice_model',
        'mnlogit_full_y'. The first two are statsmodels BinaryResultsWrapper objects (logit),
        the third is a statsmodels MNLogit results wrapper (or None).

    Returns
    -------
    dict
        A dictionary with keys:
         - "object": a nested dict with extracted coefficients, SEs, p-values, 95% CIs,
                     odds ratios (for the two binary logit models) for Age_c main effect
                     and any Age_c:C(culture) interaction terms; for mnlogit, the Age_c
                     coefficient for each non-baseline outcome (with SE, p, CI, exp(CI)).
         - "description": A concise explanation of what the extracted numbers mean
                          and how to interpret them with respect to the research question.
    """
    import numpy as np
    import pandas as pd
    import math

    def roundf(x, nd=4):
        try:
            return float(np.round(x, nd))
        except Exception:
            return x

    def summarize_binary_logit(model):
        """
        Extract Age_c main effect and Age_c * culture interaction terms from a binary logit model.
        Returns a dict with entries for 'Age_c' (if present) and 'interactions' (list).
        Each entry contains coef, se, p, ci_lower, ci_upper, odds_ratio, or_ci_lower, or_ci_upper.
        """
        if model is None:
            return None
        res = {}
        try:
            params = model.params  # Series
        except Exception:
            return None
        bse = getattr(model, 'bse', None)
        pvals = getattr(model, 'pvalues', None)
        try:
            ci_df = model.conf_int()
        except Exception:
            ci_df = None

        # Main Age_c effect
        if 'Age_c' in params.index:
            coef = params.loc['Age_c']
            se = bse.loc['Age_c'] if (bse is not None and 'Age_c' in getattr(bse, 'index', [])) else None
            p = pvals.loc['Age_c'] if (pvals is not None and 'Age_c' in getattr(pvals, 'index', [])) else None
            if ci_df is not None and 'Age_c' in getattr(ci_df, 'index', []):
                ci_low, ci_high = ci_df.loc['Age_c'].tolist()
            else:
                ci_low, ci_high = None, None
            or_val = math.exp(coef) if coef is not None and np.isfinite(coef) else None
            or_ci_low = math.exp(ci_low) if ci_low is not None else None
            or_ci_high = math.exp(ci_high) if ci_high is not None else None
            res['Age_c'] = {
                'coef': roundf(coef),
                'se': roundf(se) if se is not None else None,
                'p_value': roundf(p) if p is not None else None,
                'ci_95': (roundf(ci_low) if ci_low is not None else None,
                          roundf(ci_high) if ci_high is not None else None),
                'odds_ratio': roundf(or_val) if or_val is not None else None,
                'or_95': (roundf(or_ci_low) if or_ci_low is not None else None,
                          roundf(or_ci_high) if or_ci_high is not None else None),
            }

        # Interaction terms: any parameter name that contains "Age_c" and also "C(culture)" or ":"
        interactions = []
        for name in params.index:
            if name == 'Age_c':
                continue
            # typical statsmodels interaction naming from formula 'Age_c * C(culture)' is
            # something like 'Age_c:C(culture)[T.siteB]' or 'Age_c:C(culture)[T.<level>]'.
            # We check string membership safely by converting to str
            sname = str(name)
            if ('Age_c' in sname) and ('C(culture)' in sname or ':' in sname):
                coef = params.loc[name]
                se = bse.loc[name] if (bse is not None and name in getattr(bse, 'index', [])) else None
                p = pvals.loc[name] if (pvals is not None and name in getattr(pvals, 'index', [])) else None
                if ci_df is not None and name in getattr(ci_df, 'index', []):
                    ci_low, ci_high = ci_df.loc[name].tolist()
                else:
                    ci_low, ci_high = None, None
                or_val = math.exp(coef) if coef is not None and np.isfinite(coef) else None
                or_ci_low = math.exp(ci_low) if ci_low is not None else None
                or_ci_high = math.exp(ci_high) if ci_high is not None else None
                interactions.append({
                    'term': sname,
                    'coef': roundf(coef),
                    'se': roundf(se) if se is not None else None,
                    'p_value': roundf(p) if p is not None else None,
                    'ci_95': (roundf(ci_low) if ci_low is not None else None,
                              roundf(ci_high) if ci_high is not None else None),
                    'odds_ratio': roundf(or_val) if or_val is not None else None,
                    'or_95': (roundf(or_ci_low) if or_ci_low is not None else None,
                              roundf(or_ci_high) if or_ci_high is not None else None),
                })
        res['interactions'] = interactions
        return res

    def _find_column_key_by_name_like(columns, target_substr):
        """
        Given an iterable of column keys (which may not be strings), return the first column key
        whose string representation contains target_substr, or None.
        """
        for col in columns:
            try:
                if target_substr == str(col) or target_substr in str(col):
                    return col
            except Exception:
                continue
        return None

    def summarize_mnlogit(mn_model):
        """
        Extract Age_c coefficient (and SE, p, CI) for each non-baseline outcome in an MNLogit model.
        mn_model.params is typically a DataFrame with rows = outcomes, columns = exog names.
        """
        if mn_model is None:
            return None
        try:
            params_df = mn_model.params  # DataFrame: index = outcome categories, columns = exog names
        except Exception:
            return None
        out = {}

        # Find Age_c column key (original dtype) robustly
        age_col_key = None
        # First try exact match among columns using string comparison
        for col in params_df.columns:
            try:
                if str(col) == 'Age_c':
                    age_col_key = col
                    break
            except Exception:
                continue
        if age_col_key is None:
            # find any column whose string contains 'Age_c'
            age_col_key = _find_column_key_by_name_like(params_df.columns, 'Age_c')
        if age_col_key is None:
            # no Age_c-like column found
            return None

        # bse and pvalues may be DataFrames as well
        bse_df = None
        p_df = None
        try:
            bse_df = mn_model.bse
        except Exception:
            bse_df = None
        try:
            p_df = mn_model.pvalues
        except Exception:
            p_df = None
        try:
            ci = mn_model.conf_int()
        except Exception:
            ci = None

        for outcome in params_df.index:
            try:
                coef = params_df.loc[outcome, age_col_key]
            except Exception:
                coef = None
            se = None
            p = None
            ci_low, ci_high = None, None

            # locate matching column key in bse_df/p_df if they exist
            if bse_df is not None:
                try:
                    # bse_df may have same structure as params_df: index=outcome, columns=params
                    if (getattr(bse_df, 'index', None) is not None) and (getattr(bse_df, 'columns', None) is not None):
                        bse_col_key = _find_column_key_by_name_like(bse_df.columns, str(age_col_key))
                        if bse_col_key is None:
                            bse_col_key = _find_column_key_by_name_like(bse_df.columns, 'Age_c')
                        if bse_col_key is not None and outcome in bse_df.index:
                            se = bse_df.loc[outcome, bse_col_key]
                except Exception:
                    se = None

            if p_df is not None:
                try:
                    p_col_key = _find_column_key_by_name_like(p_df.columns, str(age_col_key))
                    if p_col_key is None:
                        p_col_key = _find_column_key_by_name_like(p_df.columns, 'Age_c')
                    if p_col_key is not None and outcome in p_df.index:
                        p = p_df.loc[outcome, p_col_key]
                except Exception:
                    p = None

            # conf_int: try to extract appropriate CI for outcome & param
            try:
                if ci is not None:
                    if isinstance(ci, pd.DataFrame) and isinstance(ci.index, pd.MultiIndex):
                        # index like (outcome, param)
                        if (outcome, age_col_key) in ci.index:
                            ci_low, ci_high = ci.loc[(outcome, age_col_key)].tolist()
                        else:
                            # try matching by string representation
                            for idx in ci.index:
                                try:
                                    if str(idx[0]) == str(outcome) and (str(idx[1]) == str(age_col_key) or 'Age_c' in str(idx[1])):
                                        ci_low, ci_high = ci.loc[idx].tolist()
                                        break
                                except Exception:
                                    continue
                    elif isinstance(ci, dict):
                        # maybe ci[outcome] is a DataFrame with params
                        if outcome in ci:
                            try:
                                ci_df_for_outcome = ci[outcome]
                                col_key = _find_column_key_by_name_like(getattr(ci_df_for_outcome, 'index', []), str(age_col_key))
                                if col_key is None:
                                    col_key = _find_column_key_by_name_like(getattr(ci_df_for_outcome, 'index', []), 'Age_c')
                                if col_key is not None:
                                    # ci[outcome] may index params by name
                                    val = ci_df_for_outcome.loc[col_key]
                                    if hasattr(val, '__iter__'):
                                        ci_low, ci_high = val.tolist()
                            except Exception:
                                pass
                    else:
                        # try ci.loc[outcome, age_col_key] if shaped appropriately
                        try:
                            # Some versions return a DataFrame with a MultiIndex columns or similar; try robust access
                            possible_col_key = _find_column_key_by_name_like(getattr(ci, 'columns', []), str(age_col_key))
                            if possible_col_key is not None and outcome in getattr(ci, 'index', []):
                                val = ci.loc[outcome, possible_col_key]
                                if hasattr(val, '__iter__'):
                                    ci_low, ci_high = list(val)
                            else:
                                # fallback: try direct loc with original keys
                                val = ci.loc[(outcome, age_col_key)] if (getattr(ci, 'index', None) is not None and (outcome, age_col_key) in ci.index) else None
                                if val is not None and hasattr(val, '__iter__'):
                                    ci_low, ci_high = list(val)
                        except Exception:
                            pass
            except Exception:
                pass

            or_val = math.exp(coef) if coef is not None and np.isfinite(coef) else None
            or_ci_low = math.exp(ci_low) if ci_low is not None else None
            or_ci_high = math.exp(ci_high) if ci_high is not None else None

            out[str(outcome)] = {
                'coef': roundf(coef),
                'se': roundf(se) if se is not None else None,
                'p_value': roundf(p) if p is not None else None,
                'ci_95': (roundf(ci_low) if ci_low is not None else None,
                          roundf(ci_high) if ci_high is not None else None),
                'exp_coef': roundf(or_val) if or_val is not None else None,
                'exp_ci_95': (roundf(or_ci_low) if or_ci_low is not None else None,
                              roundf(or_ci_high) if or_ci_high is not None else None),
            }
        return out

    result_obj = {}

    # Binary models
    social_model = model_output.get('social_follow_model')
    majority_model = model_output.get('majority_choice_model')
    mn_model = model_output.get('mnlogit_full_y')

    result_obj['social_follow_model'] = summarize_binary_logit(social_model)
    result_obj['majority_choice_model'] = summarize_binary_logit(majority_model)
    result_obj['mnlogit_full_y'] = summarize_mnlogit(mn_model)

    # Build a short textual description explaining what these values mean
    description_lines = [
        "Returned objects contain estimated Age_c effects and any Age_c × culture interaction terms.",
        "- For the two binary-logit models ('social_follow_model' and 'majority_choice_model'), each entry includes:",
        "    * coef, se, p_value, 95% CI for the coefficient on the log-odds scale;",
        "    * odds_ratio and 95% CI (exp of coefficient and CI).",
        "  Interpretation: a positive Age_c coefficient (odds_ratio > 1) means older children have higher odds",
        "  of the outcome per one year increase in Age_c (recall Age_c is mean-centered). Interaction terms",
        "  (Age_c:C(culture)[T.level]) show how the age slope differs in that culture compared to the reference.",
        "- For the multinomial model ('mnlogit_full_y'), the Age_c coef is returned for each non-baseline outcome",
        "  (coef on the log-odds scale comparing that outcome vs. the baseline category). exp_coef is the multiplicative",
        "  change in odds of that outcome (vs baseline) per one-year increase in Age_c.",
        "",
        "How to use this output to answer the research question:",
        "  * Examine the Age_c main effect in social_follow_model to see whether, across the pooled sample,",
        "    reliance on social information changes with age (p-value, sign, OR).",
        "  * Inspect interaction entries in that model to see whether the age slope differs by culture;",
        "    significant interaction coefficients indicate differing developmental trajectories across sites.",
        "  * Repeat the same logic for majority_choice_model to see whether preference for the majority",
        "    among social followers changes with age and differs by culture.",
        "  * The multinomial results provide an alternative (robustness) perspective by modeling all three choices simultaneously.",
    ]
    description = "\n".join(description_lines)

    return {
        "object": result_obj,
        "description": description
    }