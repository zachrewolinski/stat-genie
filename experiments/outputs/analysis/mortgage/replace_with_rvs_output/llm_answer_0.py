def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of 'female' from the model_output dict
    returned by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict of extracted numeric statistics for 'female'
      - "description": short interpretation in plain language
    """
    import math

    def normal_cdf(x):
        # CDF of standard normal using erf
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def find_col(df_cols, candidates):
        # Find first candidate that appears in df_cols (case insensitive)
        lc = [c.lower() for c in df_cols]
        for cand in candidates:
            if cand.lower() in lc:
                return list(df_cols)[lc.index(cand.lower())]
        return None

    res = model_output.get('model_results', None)
    odds = model_output.get('odds_ratios', None)
    conf = model_output.get('odds_ratio_conf_int', None)
    marg = model_output.get('marginal_effects', None)

    result = {}
    # Extract from model_results if available
    if res is not None:
        try:
            coef = float(res.params['female'])
            se = float(res.bse['female'])
            pval = float(res.pvalues['female'])
            result.update({
                'log_odds_coef': coef,
                'std_err': se,
                'p_value': pval,
                'significant_0.05': bool(pval < 0.05)
            })
        except Exception:
            # try attribute access by name if above fails
            try:
                params = getattr(res, 'params', {})
                bse = getattr(res, 'bse', {})
                pvalues = getattr(res, 'pvalues', {})
                coef = float(params.get('female'))
                se = float(bse.get('female'))
                pval = float(pvalues.get('female'))
                result.update({
                    'log_odds_coef': coef,
                    'std_err': se,
                    'p_value': pval,
                    'significant_0.05': bool(pval < 0.05)
                })
            except Exception:
                pass

    # Odds ratio and its CI
    if odds is not None:
        try:
            or_f = float(odds['female'])
            result['odds_ratio'] = or_f
        except Exception:
            try:
                # odds may be a pandas Series with index
                or_f = float(odds.loc['female'])
                result['odds_ratio'] = or_f
            except Exception:
                pass

    if conf is not None:
        try:
            # conf may be a DataFrame with columns [0,1] or named
            if 'female' in conf.index:
                low = float(conf.loc['female'].iloc[0])
                high = float(conf.loc['female'].iloc[1])
                result['odds_ratio_95ci'] = (low, high)
            else:
                # try label-based access
                low = float(conf.loc['female', 0])
                high = float(conf.loc['female', 1])
                result['odds_ratio_95ci'] = (low, high)
        except Exception:
            # fallback: try .at or positional
            try:
                low = float(conf.at['female', conf.columns[0]])
                high = float(conf.at['female', conf.columns[1]])
                result['odds_ratio_95ci'] = (low, high)
            except Exception:
                pass

    # Marginal effects
    if marg is not None:
        try:
            # marg is a DataFrame-like object
            # find columns for dy/dx, Std. Err., and confidence bounds (names vary)
            cols = list(marg.columns)
            dy_col = find_col(cols, ['dy/dx', 'dy_dx', 'dy/dx '])
            se_col = find_col(cols, ['Std. Err.', 'Std. Err', 'std err', 'std. err.', 'Std. Err'])
            low_col = find_col(cols, ['Conf. Int. Low', 'Conf. Int. Low ', 'Conf. Int. Low', 'conf. int. low', 'Conf. Int. Low'])
            high_col = find_col(cols, ['Conf. Int. Hi.', 'Cont. Int. Hi.', 'Conf. Int. Hi', 'conf. int. hi'])

            me = float(marg.loc['female', dy_col]) if dy_col is not None else None
            me_se = float(marg.loc['female', se_col]) if se_col is not None else None

            me_ci = None
            try:
                if low_col and high_col:
                    low = float(marg.loc['female', low_col])
                    high = float(marg.loc['female', high_col])
                    me_ci = (low, high)
            except Exception:
                me_ci = None

            result['avg_marginal_effect'] = me
            result['avg_marginal_effect_se'] = me_se
            if me is not None and me_se is not None:
                z = me / me_se if me_se != 0 else float('nan')
                p_me = 2.0 * (1.0 - normal_cdf(abs(z)))
                result['avg_marginal_effect_pvalue'] = p_me
                result['avg_marginal_effect_significant_0.05'] = bool(p_me < 0.05)
            if me_ci is not None:
                result['avg_marginal_effect_95ci'] = me_ci
        except Exception:
            pass

    # Build a human-readable description/interpretation
    desc_parts = []
    if 'odds_ratio' in result and 'odds_ratio_95ci' in result:
        desc_parts.append(
            "Controlling for the listed covariates, the estimated odds ratio for female vs male is "
            f"{result['odds_ratio']:.3f} (95% CI {result['odds_ratio_95ci'][0]:.3f} to {result['odds_ratio_95ci'][1]:.3f})."
        )
        if result.get('significant_0.05', None) or ('avg_marginal_effect_significant_0.05' in result and result['avg_marginal_effect_significant_0.05']):
            desc_parts.append("This indicates a statistically significant difference at the 5% level.")
        else:
            desc_parts.append("This difference is not statistically significant at the 5% level.")
    elif 'log_odds_coef' in result:
        desc_parts.append(
            "The model coefficient (log-odds) for female is "
            f"{result['log_odds_coef']:.3f} with p-value {result.get('p_value', float('nan')):.3g}."
        )

    if 'avg_marginal_effect' in result:
        me = result['avg_marginal_effect']
        me_ci = result.get('avg_marginal_effect_95ci', None)
        me_p = result.get('avg_marginal_effect_pvalue', None)
        if me is not None:
            if me_ci is not None:
                desc_parts.append(
                    f"The average marginal effect is about {me:.3f} (i.e. ~{me*100:.2f} percentage points), "
                    f"95% CI [{me_ci[0]:.3f}, {me_ci[1]:.3f}], p = {me_p:.3g}."
                )
            else:
                desc_parts.append(
                    f"The average marginal effect is about {me:.3f} (i.e. ~{me*100:.2f} percentage points), p = {me_p:.3g}."
                )

    if not desc_parts:
        desc = "Could not reliably extract statistics for 'female' from the provided model output."
    else:
        desc = " ".join(desc_parts) + " Covariates controlled: female (indicator), black, self_employed, married, bad_history, denied_PMI, mortgage_credit_std, consumer_credit_std, PI_ratio_std, housing_expense_ratio_std, loan_to_value_std (if present)."

    return {
        "object": result,
        "description": desc
    }