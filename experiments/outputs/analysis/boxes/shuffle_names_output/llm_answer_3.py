def extract_final_answer(model_output):
    """
    Extracts age-related coefficients, their statistics, and a joint test of
    age-by-site interactions from the two fitted GLM results provided in
    model_output (expected keys: 'model_reliance', 'model_majority').

    Returns a dict with:
      - "object": nested dict with numeric results for each model
      - "description": brief interpretation of what the extracted numbers mean
    """
    import numpy as np
    import pandas as pd

    def safe_conf_int(res):
        # conf_int may return ndarray or DataFrame; normalize to DataFrame
        try:
            ci = res.conf_int()
        except Exception:
            # if conf_int not available, return NaNs
            idx = res.params.index if hasattr(res, 'params') else []
            return pd.DataFrame(np.nan, index=idx, columns=['2.5%', '97.5%'])
        # If ndarray, convert to DataFrame using params index
        if isinstance(ci, (list, tuple)) or hasattr(ci, 'shape') and getattr(ci, 'ndim', 1) == 2 and not isinstance(ci, pd.DataFrame):
            try:
                ci_df = pd.DataFrame(ci, index=res.params.index, columns=['2.5%', '97.5%'])
            except Exception:
                # fallback: create DataFrame without index
                ci_df = pd.DataFrame(ci, columns=['2.5%', '97.5%'])
        else:
            ci_df = pd.DataFrame(ci)
            # ensure columns are named consistently if possible
            if ci_df.shape[1] >= 2:
                ci_df = ci_df.iloc[:, :2]
                ci_df.columns = ['2.5%', '97.5%']
        return ci_df

    def get_pval_from_test(test_res):
        # statsmodels WaldTestResults may store p-value as .pvalue or .pval
        if test_res is None:
            return np.nan
        for attr in ('pvalue', 'pval', 'pv'):
            if hasattr(test_res, attr):
                return float(getattr(test_res, attr))
        # try .result if present
        try:
            return float(test_res.result.pvalue)
        except Exception:
            return np.nan

    def summarize_model(res):
        out = {}
        if res is None:
            return out
        params = res.params.copy()
        bse = getattr(res, 'bse', pd.Series(np.nan, index=params.index))
        pvals = getattr(res, 'pvalues', pd.Series(np.nan, index=params.index))
        ci_df = safe_conf_int(res)

        # Find terms involving Age_c (main + interactions)
        age_terms = [n for n in params.index if 'Age_c' in n]
        age_summary = {}
        for name in age_terms:
            coef = float(params.loc[name])
            se = float(bse.loc[name]) if name in bse.index else np.nan
            p = float(pvals.loc[name]) if name in pvals.index else np.nan
            # get CI if present
            if name in ci_df.index:
                ci_lower = float(ci_df.loc[name, '2.5%'])
                ci_upper = float(ci_df.loc[name, '97.5%'])
            else:
                ci_lower, ci_upper = (np.nan, np.nan)
            age_summary[name] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                'ci_2.5%': ci_lower,
                'ci_97.5%': ci_upper,
                'significant_0.05': (not np.isnan(p)) and (p < 0.05),
                'interpretation': (
                    "Positive -> increases with age; Negative -> decreases with age"
                )
            }

        out['age_terms'] = age_summary

        # Joint test: are the age-by-site interaction terms jointly zero?
        # Interaction terms are those age_terms excluding the plain 'Age_c' (which is the reference site's slope)
        interaction_terms = [n for n in age_terms if n != 'Age_c']
        if len(interaction_terms) == 0:
            out['interaction_test'] = {
                'tested_terms': [],
                'joint_test_p_value': np.nan,
                'joint_test_significant_0.05': False,
                'note': 'No Age_c:SiteID interaction terms present in this model (maybe only one site or SiteID not factor-coded).'
            }
        else:
            # Build constraint string like "term1 = 0, term2 = 0, ..."
            constraint = ', '.join([f"{t} = 0" for t in interaction_terms])
            try:
                wres = res.wald_test(constraint)
                p_joint = get_pval_from_test(wres)
                out['interaction_test'] = {
                    'tested_terms': interaction_terms,
                    'constraint': constraint,
                    'joint_test_p_value': p_joint,
                    'joint_test_significant_0.05': (not np.isnan(p_joint)) and (p_joint < 0.05)
                }
            except Exception as e:
                out['interaction_test'] = {
                    'tested_terms': interaction_terms,
                    'constraint': constraint,
                    'joint_test_p_value': np.nan,
                    'joint_test_significant_0.05': False,
                    'note': f'Wald test failed: {e}'
                }

        return out

    result_obj = {}
    # Accept both dict-like and object with attributes
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'model_reliance' and 'model_majority'")

    # Process both models if present
    for key in ('model_reliance', 'model_majority'):
        res = model_output.get(key, None)
        if res is None:
            result_obj[key] = None
        else:
            result_obj[key] = summarize_model(res)

    # Build a human-readable description explaining what the extracted numbers mean
    desc_lines = []
    desc_lines.append("What is returned:")
    desc_lines.append("- For each model (model_reliance and model_majority) the function extracts:")
    desc_lines.append("  * Coefficients, SEs, p-values, and 95% CIs for all terms involving 'Age_c' (the age slope).")
    desc_lines.append("  * A joint Wald test p-value testing whether all age-by-site interaction terms are jointly zero.")
    desc_lines.append("")
    desc_lines.append("How to interpret the numbers:")
    desc_lines.append("- The 'Age_c' coefficient is the estimated effect of age (slope) for the reference site.")
    desc_lines.append("- Interaction terms (e.g., 'Age_c:C(SiteID)[T.X]' or 'C(SiteID)[T.X]:Age_c') show how the age slope differs for other sites relative to the reference site.")
    desc_lines.append("- If an age coefficient is positive and statistically significant (p < 0.05), reliance/preference increases with age at that site (or for the reference site if it's 'Age_c').")
    desc_lines.append("- The joint test p-value indicates whether developmental trajectories (age slopes) differ across sites overall. A significant joint p-value (p < 0.05) means age effects vary across cultural sites.")
    desc_lines.append("")
    desc_lines.append("Returned 'object' structure:")
    desc_lines.append("- object = { 'model_reliance': { 'age_terms': {...}, 'interaction_test': {...} }, 'model_majority': { ... } }")
    desc_lines.append("  Each age_term entry contains coef, se, p_value, ci_2.5%, ci_97.5%, and a boolean 'significant_0.05'.")

    description = '\n'.join(desc_lines)

    return {'object': result_obj, 'description': description}