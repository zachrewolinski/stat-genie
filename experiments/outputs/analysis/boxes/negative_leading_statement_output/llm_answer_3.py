def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether children's reliance on majority preference
    develops with age differently across cultures.

    Input:
        model_output: dict as returned by the modeling function. Expected keys:
            'mnlogit', 'logit_social', 'logit_majority_among_dem' (each a statsmodels result wrapper)
            or error keys indicating failure.

    Returns:
        dict with keys:
          - "object": a dict with extracted tables and boolean tests for:
                * culture main effects (any site differences)
                * age-by-culture interactions (differences in developmental trajectories)
            The tables contain coefficients, p-values and 95% CIs for relevant predictors.
          - "description": brief plain-language interpretation of those tests.
    """
    import pandas as pd
    import numpy as np

    out = {"mnlogit": None, "logit_social": None, "logit_majority_among_dem": None}
    summary_flags = {
        "mnlogit": {"culture_significant": False, "age_by_culture_significant": False},
        "logit_social": {"culture_significant": False, "age_by_culture_significant": False},
        "logit_majority_among_dem": {"culture_significant": False, "age_by_culture_significant": False}
    }

    def _is_interaction_name(name):
        # Treat a predictor as an age-by-culture interaction if it contains age_c and culture_
        return ('age_c' in name) and ('culture_' in name)

    def _is_culture_name(name):
        return 'culture_' in name and not _is_interaction_name(name)

    def _result_to_long_table(res):
        """
        Convert statsmodels result's params, pvalues, conf_int into a long DataFrame:
        columns = ['outcome', 'predictor', 'coef', 'pval', 'ci_low', 'ci_high']
        For binary models outcome will be None (or '1'), for multinomial there will be outcome levels.
        """
        params = res.params
        pvals = res.pvalues
        conf = res.conf_int()

        # Normalize shapes into DataFrames where rows are predictors and columns are outcomes (or single column)
        if isinstance(params, pd.Series):
            # binary/logit-like: single column
            coef_df = params.to_frame(name='coef')
            pval_df = pvals.to_frame(name='pval')
            # conf is DataFrame with two columns but indexed by predictor
            if isinstance(conf, pd.DataFrame) and conf.shape[1] == 2 and conf.index.equals(coef_df.index):
                ci_low = conf.iloc[:, 0]
                ci_high = conf.iloc[:, 1]
            else:
                # fallback: try to align
                ci_low = pd.Series(index=coef_df.index, dtype=float)
                ci_high = pd.Series(index=coef_df.index, dtype=float)
            # build long table
            long = pd.DataFrame({
                'outcome': [None] * len(coef_df),
                'predictor': coef_df.index,
                'coef': coef_df['coef'].values,
                'pval': pval_df['pval'].values,
                'ci_low': ci_low.values,
                'ci_high': ci_high.values
            })
            return long
        else:
            # params is DataFrame. Determine orientation:
            # If rows look like predictors -> keep; else transpose
            predictor_candidates = {'const', 'age_c', 'gender_male'}
            rows_look_like_predictors = any((str(r) in predictor_candidates) or str(r).startswith('culture_') or 'age_c' in str(r) for r in params.index)
            if not rows_look_like_predictors:
                coef_df = params.T.copy()
                pval_df = pvals.T.copy()
                conf_df = conf.T.copy()
            else:
                coef_df = params.copy()
                pval_df = pvals.copy()
                conf_df = conf.copy()

            # conf_df may have MultiIndex columns if returned differently; handle common case:
            # conf_df columns are [0,1] with same index as coef_df
            long_rows = []
            for outcome in coef_df.columns:
                for predictor in coef_df.index:
                    coef = coef_df.at[predictor, outcome]
                    pval = pval_df.at[predictor, outcome] if predictor in pval_df.index and outcome in pval_df.columns else np.nan
                    # conf: try to get conf_df.loc[predictor] -> might be a Series with MultiIndex outcome
                    ci_low = np.nan
                    ci_high = np.nan
                    try:
                        # conf_df may be shaped like coef_df (rows predictors, cols MultiIndex with outcome)
                        if (predictor in conf_df.index) and (outcome in conf_df.columns):
                            ci_pair = conf_df.at[predictor, outcome]
                            # often conf_df.at returns array-like if columns multi-leveled; handle simple case below
                        # try simpler access: conf_df.loc[predictor] might be DataFrame with columns = outcomes (2-level)
                        # attempt to index by (predictor, outcome)
                        # Best practical approach: compute conf_int from params and bse if available:
                    except Exception:
                        pass
                    # fallback: compute ci from coef and bse if possible
                    ci_low = np.nan
                    ci_high = np.nan
                    try:
                        bse = res.bse
                        if isinstance(bse, pd.Series):
                            b = bse.get(predictor, np.nan)
                            if not np.isnan(b):
                                ci_low = coef - 1.96 * b
                                ci_high = coef + 1.96 * b
                        else:
                            # DataFrame case
                            # ensure orientation matches coef_df
                            if not rows_look_like_predictors:
                                b = bse.T.at[predictor, outcome]
                            else:
                                b = bse.at[predictor, outcome]
                            ci_low = coef - 1.96 * b
                            ci_high = coef + 1.96 * b
                    except Exception:
                        ci_low = np.nan
                        ci_high = np.nan

                    long_rows.append({
                        'outcome': outcome,
                        'predictor': predictor,
                        'coef': coef,
                        'pval': pval,
                        'ci_low': ci_low,
                        'ci_high': ci_high
                    })
            long = pd.DataFrame(long_rows)
            return long

    # Process each model if present
    for key in ['mnlogit', 'logit_social', 'logit_majority_among_dem']:
        res = model_output.get(key)
        if res is None:
            out[key] = {"error": "model not present in model_output"}
            continue
        # If an error string was returned instead of a model:
        if isinstance(res, str):
            out[key] = {"error": res}
            continue

        try:
            long_tbl = _result_to_long_table(res)

            # Identify culture main effects and age-by-culture interactions and age main effect
            culture_rows = long_tbl[long_tbl['predictor'].apply(_is_culture_name)].copy()
            interaction_rows = long_tbl[long_tbl['predictor'].apply(_is_interaction_name)].copy()
            age_rows = long_tbl[long_tbl['predictor'] == 'age_c'].copy()

            # Determine significance: any p < .05 among the relevant predictors (across outcomes)
            def any_significant(df):
                if df is None or df.shape[0] == 0:
                    return False
                pvals = pd.to_numeric(df['pval'], errors='coerce')
                return (pvals < 0.05).any()

            culture_sig = any_significant(culture_rows)
            interaction_sig = any_significant(interaction_rows)

            summary_flags[key]['culture_significant'] = bool(culture_sig)
            summary_flags[key]['age_by_culture_significant'] = bool(interaction_sig)

            out[key] = {
                'long_table': long_tbl,
                'culture_effects': culture_rows.reset_index(drop=True),
                'interaction_effects': interaction_rows.reset_index(drop=True),
                'age_effect': age_rows.reset_index(drop=True),
                'culture_significant': bool(culture_sig),
                'age_by_culture_significant': bool(interaction_sig)
            }
        except Exception as e:
            out[key] = {"error": f"failed to extract from model: {e}"}

    # Construct a concise description interpreting the critical tests
    desc_lines = []
    # For the primary multinomial model
    mn = out.get('mnlogit')
    if isinstance(mn, dict) and 'error' not in mn:
        cs = mn['culture_significant']
        ia = mn['age_by_culture_significant']
        if cs and ia:
            desc_lines.append("Multinomial model: There is evidence of (a) cross-cultural differences in choice probabilities and (b) age-by-culture interactions (developmental trajectories differ across sites).")
        elif cs and not ia:
            desc_lines.append("Multinomial model: There are cross-cultural differences in choice probabilities, but no evidence that age-related trajectories differ across cultures.")
        elif (not cs) and ia:
            desc_lines.append("Multinomial model: No overall cross-cultural differences in choice probabilities, but there is evidence that age-related trajectories differ across cultures.")
        else:
            desc_lines.append("Multinomial model: No evidence that choice probabilities differ across cultures, and no evidence that age-related trajectories differ across cultures.")
    else:
        desc_lines.append("Multinomial model: results not available or extraction failed.")

    # For binary social_choice model
    soc = out.get('logit_social')
    if isinstance(soc, dict) and 'error' not in soc:
        cs = soc['culture_significant']
        ia = soc['age_by_culture_significant']
        if cs and ia:
            desc_lines.append("Binary (social_choice) model: Evidence of cross-cultural differences and differing age trajectories across cultures.")
        elif cs and not ia:
            desc_lines.append("Binary (social_choice) model: Cross-cultural differences present, but developmental (age) trajectories seem similar across cultures.")
        elif (not cs) and ia:
            desc_lines.append("Binary (social_choice) model: No overall cross-cultural differences, but age trajectories vary by culture.")
        else:
            desc_lines.append("Binary (social_choice) model: No cross-cultural differences and no evidence that age trajectories differ across cultures.")
    else:
        desc_lines.append("Binary (social_choice) model: results not available or extraction failed.")

    # For majority_among_dem model
    maj = out.get('logit_majority_among_dem')
    if isinstance(maj, dict) and 'error' not in maj:
        cs = maj['culture_significant']
        ia = maj['age_by_culture_significant']
        if cs and ia:
            desc_lines.append("Binary (majority among demonstrated) model: Evidence of cross-cultural differences and differing age trajectories across cultures.")
        elif cs and not ia:
            desc_lines.append("Binary (majority among demonstrated) model: Cross-cultural differences present, but developmental trajectories similar across cultures.")
        elif (not cs) and ia:
            desc_lines.append("Binary (majority among demonstrated) model: No overall cross-cultural differences, but age trajectories vary by culture.")
        else:
            desc_lines.append("Binary (majority among demonstrated) model: No cross-cultural differences and no evidence of differing age-related trajectories across cultures.")
    else:
        desc_lines.append("Binary (majority among demonstrated) model: results not available or extraction failed.")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}