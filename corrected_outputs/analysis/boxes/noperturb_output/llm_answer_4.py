def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for age-related terms
    (Age_c, Age_c2, and Age_c by culture interaction terms) from a fitted
    statsmodels MNLogit results wrapper.

    Returns a dictionary with:
      - "object": a dict containing a table of extracted statistics (one row per
                  outcome x parameter) and a concise summary focusing on the
                  Age_c effects for the majority outcome and interactions by culture.
      - "description": a human-readable explanation of what the numbers mean.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Try to get params and bse; fall back if not available
    try:
        params = res.params
        bse = res.bse
        pvals = res.pvalues
    except Exception as e:
        raise ValueError("Provided object does not appear to be a fitted statsmodels results object "
                         "with .params/.bse/.pvalues attributes.") from e

    # Normalize to a long table with columns: outcome, param, coef, bse, pval
    if isinstance(params, pd.DataFrame):
        # Typical shape for MNLogit: index = parameter names, columns = outcome labels
        coef_long = params.stack().reset_index()
        coef_long.columns = ['param', 'outcome', 'coef']
        bse_long = bse.stack().reset_index()
        bse_long.columns = ['param', 'outcome', 'bse']
        # pvalues may be DataFrame too
        if isinstance(pvals, pd.DataFrame):
            pval_long = pvals.stack().reset_index()
            pval_long.columns = ['param', 'outcome', 'pval']
        else:
            # If pvals not a DataFrame, compute from z
            z = coef_long['coef'] / bse_long['bse']
            pval_long = coef_long[['param', 'outcome']].copy()
            pval_long['pval'] = 2 * (1 - stats.norm.cdf(np.abs(z)))
        # Merge
        table = pd.merge(coef_long, bse_long, on=['param', 'outcome'])
        table = pd.merge(table, pval_long, on=['param', 'outcome'])
    else:
        # If params is ndarray: reconstruct using model exog names and outcome ordering
        try:
            param_names = res.model.exog_names
        except Exception:
            raise ValueError("Cannot determine parameter names from the model object.")
        arr = np.asarray(params)
        arr_bse = np.asarray(bse)
        # Number of non-reference outcomes is arr.shape[0] or arr.shape[1]; handle both
        if arr.ndim == 2:
            # assume shape (n_outcomes, n_params) or (n_params, n_outcomes)
            if arr.shape[1] == len(param_names):
                arr = arr  # (n_outcomes x n_params)
            elif arr.shape[0] == len(param_names):
                arr = arr.T
            else:
                # fallback
                arr = arr
        n_outcomes, n_params = arr.shape
        # label outcomes as strings '1', '2', ... (these correspond to model's non-reference categories)
        outcome_labels = [str(i) for i in range(1, n_outcomes + 1)]
        rows = []
        for i, out in enumerate(outcome_labels):
            for j, pname in enumerate(param_names):
                coef = float(arr[i, j])
                se = float(arr_bse[i, j]) if arr_bse.shape == arr.shape else float(arr_bse[j])
                z = coef / se if se != 0 else np.nan
                pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                rows.append({'param': pname, 'outcome': out, 'coef': coef, 'bse': se, 'pval': pval})
        table = pd.DataFrame(rows)

    # Compute 95% CI from coef +/- 1.96*bse
    table['ci_lower'] = table['coef'] - 1.96 * table['bse']
    table['ci_upper'] = table['coef'] + 1.96 * table['bse']

    # Focus on age-related terms:
    # - linear: 'Age_c' (but not 'Age_c2')
    # - quadratic: contains 'Age_c2' or 'Age_c2' literal
    # - interactions: parameter names containing both 'Age_c' and either ':' or 'C(culture)'
    def is_age_linear(p):
        return ('Age_c' in p) and ('Age_c2' not in p) and (':' not in p and 'C(culture)' not in p)
    def is_age_quadratic(p):
        return ('Age_c2' in p) or ('Age_c^2' in p)  # tolerant match
    def is_age_interaction(p):
        return ('Age_c' in p) and ((':' in p) or ('C(culture)' in p)) and ('Age_c2' not in p)

    table['term_type'] = table['param'].apply(
        lambda p: 'age_linear' if is_age_linear(p)
        else ('age_quadratic' if is_age_quadratic(p)
              else ('age_interaction' if is_age_interaction(p) else 'other')))

    # Summarize for outcomes of interest (we expect strings like '1' for majority and '2' for minority)
    # Try to detect which outcome corresponds to 'majority' and 'minority' by name:
    # If outcomes were numeric (1,2), we use those; otherwise we'll present all outcomes.
    outcomes = sorted(table['outcome'].unique(), key=lambda x: str(x))

    # Build a concise summary focusing on:
    # - Age linear effect on outcome '1' (majority) if present
    # - Age quadratic on outcome '1'
    # - Interactions (list) for 'Age_c' by culture across outcomes
    summary_lines = []
    def summarize_term(df_row):
        coef = df_row['coef']
        se = df_row['bse']
        p = df_row['pval']
        ci_l = df_row['ci_lower']
        ci_u = df_row['ci_upper']
        return f"coef={coef:.3f}, se={se:.3f}, p={p:.3f}, 95%CI=[{ci_l:.3f}, {ci_u:.3f}]"

    # Outcome label assumptions: try "1" as majority, "2" as minority if present
    maj_label = None
    min_label = None
    if '1' in outcomes:
        maj_label = '1'
    elif len(outcomes) >= 1:
        maj_label = outcomes[0]
    if '2' in outcomes:
        min_label = '2'
    elif len(outcomes) >= 2:
        min_label = outcomes[1]

    # Summarize age linear & quadratic for majority outcome
    if maj_label is not None:
        maj_df = table[(table['outcome'] == maj_label) & (table['term_type'].isin(['age_linear', 'age_quadratic']))]
        if not maj_df.empty:
            for _, row in maj_df.iterrows():
                tname = 'Age (linear)' if row['term_type'] == 'age_linear' else 'Age (quadratic)'
                summary_lines.append(f"Majority outcome (outcome={maj_label}) - {tname}: {summarize_term(row)}")
        else:
            summary_lines.append(f"No explicit Age terms found for majority outcome (outcome={maj_label}).")
    else:
        summary_lines.append("No majority outcome label could be detected in the model outcomes.")

    # Summarize age interactions by culture (for any outcome)
    inter_df = table[table['term_type'] == 'age_interaction'].copy()
    if not inter_df.empty:
        # Show top interactions
        for _, row in inter_df.iterrows():
            summary_lines.append(f"Interaction - outcome={row['outcome']}, param={row['param']}: {summarize_term(row)}")
    else:
        summary_lines.append("No Age x Culture interaction terms detected in the fitted model parameters.")

    # Interpret significance for Age linear on majority outcome if available
    interpret = ""
    if maj_label is not None:
        maj_age_linear = table[(table['outcome'] == maj_label) & (table['term_type'] == 'age_linear')]
        if not maj_age_linear.empty:
            row = maj_age_linear.iloc[0]
            p = row['pval']
            if p < 0.05:
                interpret = ("The linear age effect on choosing the majority option (relative to the "
                             "reference category) is statistically significant (p < 0.05). "
                             "The coefficient reported is on the log-odds scale: a positive value "
                             "means older children are more likely to choose the majority option.")
            else:
                interpret = ("The linear age effect on choosing the majority option is not statistically significant "
                             "(p >= 0.05). There is no strong evidence of a linear increase/decrease in reliance on the majority "
                             "with age in the reference culture implied by the model.")
        else:
            interpret = "No linear Age_c parameter found for the majority outcome; cannot assess a simple linear age effect."
    else:
        interpret = "Could not identify which outcome corresponds to the majority choice to make a direct interpretation."

    # Prepare the object to return: convert table to records for portability
    table_records = table.sort_values(['outcome', 'param']).to_dict(orient='records')

    result_object = {
        'table': table_records,
        'summary_lines': summary_lines,
        'interpretation': interpret,
        'notes': [
            "Coefficients are log-odds (logits) for choosing the given outcome vs the reference category (y_mn==0).",
            "Age_c is mean-centered age; Age_c2 is quadratic term. Interaction parameters denote deviations of the age slope for non-reference cultures.",
            "Significance is based on p-values (two-sided). CIs shown are approximate 95% (coef ± 1.96*SE)."
        ]
    }

    description = (
        "Extracted parameter estimates, standard errors, p-values, and 95% confidence intervals for all model parameters. "
        "I highlighted age-related terms (linear, quadratic, and interactions with culture) and provided a brief interpretation "
        "about whether there is evidence that reliance on the majority option changes with age (particularly for the majority outcome). "
        "The coefficients are in log-odds: positive linear Age_c for the majority outcome means older children become more likely to choose the majority option "
        "(relative to the reference/unselected option). Interaction terms indicate whether the age slope differs by cultural site."
    )

    return {'object': result_object, 'description': description}