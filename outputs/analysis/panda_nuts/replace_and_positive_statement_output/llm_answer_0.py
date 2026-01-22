def extract_final_answer(model_output):
    """
    Extracts coefficients, incidence-rate ratios (IRRs), clustered CIs, and p-values
    for age_c, sex_M, and help_Y from the model_output returned by the modeling function.
    Returns a dict with:
      - "object": a pandas DataFrame summarizing coef, IRR, CI, p-value, and significance
      - "description": plain-language interpretation of the results in context

    The function is robust to two shapes of model_output:
      - The exact dict shown in the prompt (with 'final_results_clustered' and/or 'irr_table')
      - A more generic dict with a statsmodels results-like object that provides params, pvalues, conf_int
    """
    import pandas as pd
    import numpy as np

    vars_of_interest = ['age_c', 'sex_M', 'help_Y']
    # Prepare an empty results container
    rows = []

    # Helper to safely extract values from a table-like object
    def safe_get(df_like, idx, col, default=np.nan):
        try:
            return df_like.loc[idx, col]
        except Exception:
            try:
                # maybe it's a dict-like row
                return df_like[idx][col]
            except Exception:
                return default

    # Try using irr_table if available (contains IRR and clustered p-values)
    irr_table = model_output.get('irr_table') if isinstance(model_output, dict) else None
    final_res = model_output.get('final_results_clustered') if isinstance(model_output, dict) else None

    if isinstance(irr_table, pd.DataFrame):
        # Use the irr_table rows when present
        for v in vars_of_interest:
            if v in irr_table.index:
                coef = float(irr_table.loc[v, 'coef'])
                irr = float(irr_table.loc[v, 'IRR'])
                ci_l = float(irr_table.loc[v, 'IRR_CI_lower'])
                ci_u = float(irr_table.loc[v, 'IRR_CI_upper'])
                p = float(irr_table.loc[v, 'pvalue'])
                rows.append({'term': v, 'coef': coef, 'IRR': irr, 'IRR_CI_lower': ci_l, 'IRR_CI_upper': ci_u, 'pvalue': p})
            else:
                rows.append({'term': v, 'coef': np.nan, 'IRR': np.nan, 'IRR_CI_lower': np.nan, 'IRR_CI_upper': np.nan, 'pvalue': np.nan})
    elif final_res is not None:
        # final_res is expected to have params, pvalues, and conf_int()
        params = getattr(final_res, 'params', None)
        pvalues = getattr(final_res, 'pvalues', None)
        try:
            conf = final_res.conf_int()
        except Exception:
            conf = None

        for v in vars_of_interest:
            try:
                coef = float(params[v]) if params is not None and v in params.index else np.nan
            except Exception:
                coef = np.nan
            try:
                p = float(pvalues[v]) if pvalues is not None and v in pvalues.index else np.nan
            except Exception:
                p = np.nan
            # Confidence intervals in coefficient scale
            if conf is not None and v in conf.index:
                ci_l = float(conf.loc[v, 0])
                ci_u = float(conf.loc[v, 1])
                irr_ci_l = float(np.exp(ci_l))
                irr_ci_u = float(np.exp(ci_u))
            else:
                ci_l = ci_u = irr_ci_l = irr_ci_u = np.nan

            irr = float(np.exp(coef)) if not np.isnan(coef) else np.nan

            rows.append({'term': v, 'coef': coef, 'IRR': irr, 'IRR_CI_lower': irr_ci_l, 'IRR_CI_upper': irr_ci_u, 'pvalue': p})
    else:
        # As a last resort, try to interpret model_output itself if it's a results-like object
        # (less likely given the prompt, but included for robustness)
        mo = model_output
        params = getattr(mo, 'params', None)
        pvalues = getattr(mo, 'pvalues', None)
        try:
            conf = mo.conf_int()
        except Exception:
            conf = None
        for v in vars_of_interest:
            try:
                coef = float(params[v]) if params is not None and v in params.index else np.nan
            except Exception:
                coef = np.nan
            try:
                p = float(pvalues[v]) if pvalues is not None and v in pvalues.index else np.nan
            except Exception:
                p = np.nan
            if conf is not None and v in conf.index:
                ci_l = float(conf.loc[v, 0])
                ci_u = float(conf.loc[v, 1])
                irr_ci_l = float(np.exp(ci_l))
                irr_ci_u = float(np.exp(ci_u))
            else:
                irr_ci_l = irr_ci_u = np.nan
            irr = float(np.exp(coef)) if not np.isnan(coef) else np.nan
            rows.append({'term': v, 'coef': coef, 'IRR': irr, 'IRR_CI_lower': irr_ci_l, 'IRR_CI_upper': irr_ci_u, 'pvalue': p})

    # Build DataFrame
    summary_df = pd.DataFrame(rows).set_index('term')
    # Add significance flag and human-friendly interpretation per term
    alpha = 0.05
    def interpret_row(r):
        p = r['pvalue']
        irr = r['IRR']
        ci_l = r['IRR_CI_lower']
        ci_u = r['IRR_CI_upper']
        if pd.isna(p):
            sig = False
            interp = "No p-value available."
        else:
            sig = (p < alpha)
            if sig:
                direction = "increase" if r['coef'] > 0 else "decrease"
                interp = f"Statistically significant ({p:.3f}). Estimated {direction} in nut-opening rate (IRR={irr:.3f}, CI {ci_l:.3f}–{ci_u:.3f})."
            else:
                interp = f"No strong evidence of an effect (p={p:.3f}). Point estimate IRR={irr:.3f} with CI {ci_l:.3f}–{ci_u:.3f}, which includes 1, so the direction is uncertain."
        return pd.Series({'significant': sig, 'interpretation': interp})

    extra = summary_df.apply(interpret_row, axis=1)
    summary_df = pd.concat([summary_df, extra], axis=1)

    # Compose a short overall description
    def make_overall_description(df):
        lines = []
        lines.append("Question: How do age, sex (male vs female), and receiving help influence nut-cracking efficiency (nuts opened per unit time)?")
        # Check whether any of the three are significant
        sig_terms = df.index[df['significant']].tolist()
        if len(sig_terms) == 0:
            lines.append("Answer (summary): There is no strong evidence (alpha = 0.05) that age, sex, or receiving help are associated with nut-cracking efficiency in this model.")
        else:
            lines.append("Answer (summary): The following predictors showed statistically significant associations with nut-cracking rate: " + ", ".join(sig_terms))
        # Add per-term brief notes
        for term in df.index:
            interp = df.loc[term, 'interpretation']
            lines.append(f"- {term}: {interp}")
        lines.append("Notes: Estimates are incidence-rate ratios (IRR) derived from a count model using session duration as exposure. Confidence intervals and p-values are clustered by chimpanzee to account for repeated sessions. Effects whose CI include 1 should be treated as uncertain.")
        return " ".join(lines)

    description = make_overall_description(summary_df)

    return {'object': summary_df, 'description': description}