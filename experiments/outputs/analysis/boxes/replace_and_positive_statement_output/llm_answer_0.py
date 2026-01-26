def extract_final_answer(model_output):
    """
    Extracts age-related statistics from two fitted GLM models returned in model_output.
    Expects model_output to be a dict with keys 'model_demonstrated' and 'model_majority'
    (statsmodels result objects).
    
    Returns a dictionary with:
      - "object": a dict summarizing coefficients, SEs, p-values, 95% CIs, odds ratios
                  for the age main effect and any age-by-culture interaction terms
                  for each model.
      - "description": a concise interpretation of what those statistics imply about
                       developmental (age) effects and whether they vary across cultures.
    """
    import numpy as np
    import pandas as pd

    out = {}
    interpretations = []

    # Helper to summarize age-related params for one model
    def summarize_model(name, model):
        # Collect basic arrays
        params = model.params
        bse = model.bse
        pvals = model.pvalues
        # conf_int may return array or DataFrame
        ci = model.conf_int()
        try:
            ci_df = pd.DataFrame(ci, index=params.index, columns=['ci_low', 'ci_high'])
        except Exception:
            # fallback: assume it's an ndarray with same index order
            ci_df = pd.DataFrame(ci, index=params.index)
            ci_df.columns = ['ci_low', 'ci_high']

        # Find all parameter names that reference age_centered
        age_param_names = [pn for pn in params.index if 'age_centered' in pn]

        if not age_param_names:
            return {
                'age_params_table': [],
                'summary': f'No parameters containing "age_centered" found in model "{name}".'
            }

        rows = []
        for pn in age_param_names:
            coef = float(params.loc[pn])
            se = float(bse.loc[pn]) if pn in bse.index else np.nan
            p = float(pvals.loc[pn]) if pn in pvals.index else np.nan
            ci_low = float(ci_df.loc[pn, 'ci_low'])
            ci_high = float(ci_df.loc[pn, 'ci_high'])
            z = coef / se if se and not np.isnan(se) else np.nan
            oratio = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            rows.append({
                'param': pn,
                'coef_logodds': coef,
                'se': se,
                'z': z,
                'pvalue': p,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'odds_ratio': oratio,
                'or_ci_95_low': or_ci_low,
                'or_ci_95_high': or_ci_high
            })

        table = pd.DataFrame(rows)

        # Interpret main age effect and interactions
        main_row = table[table['param'] == 'age_centered']
        if not main_row.empty:
            main = main_row.iloc[0]
            main_sig = main['pvalue'] < 0.05 if not np.isnan(main['pvalue']) else False
            main_dir = 'increase' if main['coef_logodds'] > 0 else ('decrease' if main['coef_logodds'] < 0 else 'no change')
            main_text = (f'Main age effect (age_centered) in model "{name}": coefficient={main["coef_logodds"]:.3f}, '
                         f'p={main["pvalue"]:.3g}. This corresponds to an odds ratio={main["odds_ratio"]:.3f} '
                         f'(95% CI [{main["or_ci_95_low"]:.3f}, {main["or_ci_95_high"]:.3f}]). '
                         f'Interpretation: a per-unit increase in age is associated with a {main_dir} '
                         f'in the log-odds of the outcome. {"(statistically significant)" if main_sig else "(not statistically significant)"}')
        else:
            main_text = f'No simple main effect parameter named "age_centered" found in model "{name}".'

        # Check interactions: any other age-related params are interactions (age_centered:C(...))
        interaction_rows = table[table['param'] != 'age_centered']
        if not interaction_rows.empty:
            sig_interactions = interaction_rows[interaction_rows['pvalue'] < 0.05]
            if not sig_interactions.empty:
                inter_text = (f'Age-by-culture interactions present and significant for the following terms in model "{name}": '
                              f'{", ".join(sig_interactions["param"].tolist())}. These indicate that the developmental '
                              f'(age) slope differs in those cultures relative to the reference culture.')
            else:
                inter_text = (f'Age-by-culture interaction terms are present in model "{name}" but none reach p<0.05; '
                              f'this suggests no strong evidence that developmental slopes differ across cultures.')
        else:
            inter_text = f'No age-by-culture interaction terms found in model "{name}".'

        summary = main_text + ' ' + inter_text

        return {
            'age_params_table': table.to_dict(orient='records'),
            'summary': summary
        }

    # Expecting two models; handle generically for any in the dict
    for key in model_output:
        try:
            model = model_output[key]
            out[key] = summarize_model(key, model)
            interpretations.append(out[key]['summary'])
        except Exception as e:
            out[key] = {'error': str(e)}
            interpretations.append(f'Could not summarize model "{key}": {e}')

    description = ("Summary of age effects and age-by-culture interactions for each model:\n" +
                   "\n".join(interpretations) +
                   "\n\nNotes: coefficients are on the log-odds scale; odds ratios = exp(coef). "
                   "A significant main age effect (p<0.05) implies a consistent developmental change in the outcome "
                   "across cultures (same slope across cultures). Significant age-by-culture interaction terms imply "
                   "that the developmental slope differs between the reference culture and the listed cultures. "
                   "Inspect the per-term coefficients and odds ratios above to see direction and magnitude for each term.")

    return {'object': out, 'description': description}