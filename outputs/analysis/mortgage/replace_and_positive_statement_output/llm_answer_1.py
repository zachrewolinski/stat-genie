def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of applicant gender ('female') on mortgage acceptance
    from the model_output produced by the provided modelling function.

    Returns a dictionary with keys:
      - "object": dict with numeric results and a short, coded conclusion
      - "description": human-readable explanation of the extracted statistics and conclusion
    """
    import numpy as np
    import pandas as pd

    result = model_output.get('model_result', None)
    odds_df = model_output.get('odds_ratios', None)
    marg_eff_df = model_output.get('marginal_effect_female', None)

    # Initialize outputs with Nones so function is robust
    coef = None
    pval = None
    ci_lower = None
    ci_upper = None
    odds_ratio = None
    odds_ci = (None, None)

    me_dx = None
    me_se = None
    me_pval = None
    me_ci_low = None
    me_ci_high = None

    # 1) Extract coefficient, p-value, and conf int from model_result if available
    if result is not None:
        try:
            # params and pvalues are usually pandas Series indexed by variable name
            coef = float(result.params.get('female', result.params.get('female', np.nan)))
        except Exception:
            coef = None
        try:
            pval = float(result.pvalues.get('female', np.nan))
        except Exception:
            pval = None
        try:
            ci = result.conf_int()
            # conf_int typically has rows indexed by variable names
            if 'female' in ci.index:
                ci_lower = float(ci.loc['female'][0])
                ci_upper = float(ci.loc['female'][1])
        except Exception:
            ci_lower = ci_upper = None

    # 2) Odds ratio and its CI (prefer odds_df if provided)
    if isinstance(odds_df, (pd.DataFrame, pd.Series)) and 'OR' in getattr(odds_df, 'columns', []):
        try:
            if 'female' in odds_df.index:
                odds_ratio = float(odds_df.loc['female', 'OR'])
                # attempt to get CI columns named '2.5%' and '97.5%' or similar
                if '2.5%' in odds_df.columns and '97.5%' in odds_df.columns:
                    odds_ci = (float(odds_df.loc['female', '2.5%']), float(odds_df.loc['female', '97.5%']))
        except Exception:
            odds_ratio = None
    else:
        # fallback: compute from coef and conf int if present
        try:
            if coef is not None:
                odds_ratio = float(np.exp(coef))
            if ci_lower is not None and ci_upper is not None:
                odds_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
        except Exception:
            odds_ratio = odds_ci = (None, None)

    # 3) Marginal effect for female (from provided marginal_effect_female if it's a DataFrame)
    if isinstance(marg_eff_df, pd.DataFrame):
        # Try to find a row for 'female' (case-insensitive)
        idx = None
        for r in marg_eff_df.index:
            if str(r).lower() == 'female':
                idx = r
                break
        if idx is None:
            # try partial match
            for r in marg_eff_df.index:
                if 'female' in str(r).lower():
                    idx = r
                    break
        if idx is not None:
            # column names vary; attempt common ones
            # common names: 'dy/dx', 'Std. Err.', 'P>|z|', 'Conf. Int. Low', 'Conf. Int. Hi'
            row = marg_eff_df.loc[idx]
            # dy/dx
            for col in ['dy/dx', 'dy_dx', 'dydx', 'dydx', 'effect', 'dy/dx  ']:
                if col in row.index:
                    me_dx = float(row[col])
                    break
            # Std. Err.
            for col in ['Std. Err.', 'std err', 'std_err', 'Std. Err', 'StdErr']:
                if col in row.index:
                    me_se = float(row[col])
                    break
            # p-value
            for col in ['P>|z|', 'pvalue', 'p-value', 'p_value', 'P>z', 'P>|z| ']:
                if col in row.index:
                    try:
                        me_pval = float(row[col])
                    except Exception:
                        me_pval = None
                    break
            # confidence interval columns
            for low_name in ['Conf. Int. Low', 'CI_low', 'ci_low', 'Conf. Int. [0.025', '0.025']:
                if low_name in row.index:
                    me_ci_low = float(row[low_name])
                    break
            for high_name in ['Conf. Int. Hi', 'CI_high', 'ci_high', 'Conf. Int. 0.975', '0.975']:
                if high_name in row.index:
                    me_ci_high = float(row[high_name])
                    break
            # If no CI columns by those names, try to infer last two columns are CI
            if me_ci_low is None or me_ci_high is None:
                # pick numeric columns at end that look like CI
                numeric = [c for c in row.index if pd.api.types.is_numeric_dtype(type(row[c])) or isinstance(row[c], (int, float, np.floating, np.integer))]
                # Already used above; fallback to named columns directly
                if 'Conf. Int. Low' in row.index and 'Conf. Int. Hi' in row.index:
                    me_ci_low = float(row['Conf. Int. Low'])
                    me_ci_high = float(row['Conf. Int. Hi'])
    else:
        # If marginal effect wasn't stored as DataFrame, attempt to compute average marginal effect via get_margeff on the result
        try:
            me_obj = result.get_margeff(at='overall', method='dydx')
            me_summary = me_obj.summary_frame()
            if 'female' in me_summary.index:
                me_row = me_summary.loc['female']
                # column names expected: 'dy/dx', 'Std. Err.', 'z', 'P>|z|', 'Conf. Int. Low', 'Conf. Int. Hi'
                me_dx = float(me_row.get('dy/dx', me_row.get('dy_dx', np.nan)))
                me_se = float(me_row.get('Std. Err.', me_row.get('std err', np.nan)))
                me_pval = float(me_row.get('P>|z|', me_row.get('pvalue', np.nan)))
                me_ci_low = float(me_row.get('Conf. Int. Low', me_row.get('0.025', np.nan)))
                me_ci_high = float(me_row.get('Conf. Int. Hi', me_row.get('0.975', np.nan)))
        except Exception:
            pass

    # 4) Decide significance and direction
    significant = False
    direction = 'none'
    # prefer coefficient p-value if available, else marginal effect p-value
    use_p = None
    if pval is not None and not np.isnan(pval):
        use_p = pval
        est_val = coef
    elif me_pval is not None and not np.isnan(me_pval):
        use_p = me_pval
        est_val = me_dx
    else:
        use_p = None
        est_val = coef if coef is not None else me_dx

    if use_p is not None:
        significant = (use_p < 0.05)
        if significant:
            if est_val is not None:
                if est_val > 0:
                    direction = 'positive'
                elif est_val < 0:
                    direction = 'negative'
                else:
                    direction = 'none'
    else:
        # If no p-value available, infer significance conservatively from CIs:
        # If odds ratio CI lies strictly above 1 -> positive significant; strictly below 1 -> negative significant
        if odds_ci[0] is not None and odds_ci[1] is not None:
            if odds_ci[0] > 1:
                significant = True
                direction = 'positive'
            elif odds_ci[1] < 1:
                significant = True
                direction = 'negative'
            else:
                significant = False
                direction = 'none'
        elif me_ci_low is not None and me_ci_high is not None:
            if me_ci_low > 0:
                significant = True
                direction = 'positive'
            elif me_ci_high < 0:
                significant = True
                direction = 'negative'
            else:
                significant = False
                direction = 'none'

    # 5) Human-readable conclusion
    if significant:
        if direction == 'positive':
            conclusion = "Being female is associated with a statistically significant higher probability of mortgage acceptance."
        elif direction == 'negative':
            conclusion = "Being female is associated with a statistically significant lower probability of mortgage acceptance."
        else:
            conclusion = "Being female is associated with a statistically significant effect on mortgage acceptance (direction unclear)."
    else:
        conclusion = ("No statistically significant effect of being female on mortgage acceptance "
                      "was found at the 0.05 level. Coefficient and marginal-effect confidence "
                      "intervals include the null (OR ~ 1, marginal effect ~ 0).")

    # Assemble the object to return
    obj = {
        'coef_female': coef,
        'p_value_female': pval,
        'coef_95ci': (ci_lower, ci_upper),
        'odds_ratio_female': odds_ratio,
        'odds_ratio_95ci': odds_ci,
        'marginal_effect_female': me_dx,
        'marginal_effect_se': me_se,
        'marginal_effect_pvalue': me_pval,
        'marginal_effect_95ci': (me_ci_low, me_ci_high),
        'significant_at_0.05': bool(significant),
        'direction': direction,
        'conclusion': conclusion
    }

    description_lines = [
        "Extracted statistics for the 'female' predictor from the fitted model:",
        f"- Logit coefficient (female): {coef}",
        f"- Coefficient p-value: {pval}",
        f"- 95% CI for coefficient: ({ci_lower}, {ci_upper})",
        f"- Odds ratio (exp(coef)) for female: {odds_ratio} with 95% CI {odds_ci}",
        f"- Average marginal effect (female): {me_dx} (SE={me_se}), 95% CI { (me_ci_low, me_ci_high) }, p={me_pval}",
        f"- Conclusion: {conclusion}"
    ]
    description = "\n".join(description_lines)

    return {'object': obj, 'description': description}