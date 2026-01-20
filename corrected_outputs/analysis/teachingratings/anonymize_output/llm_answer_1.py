def extract_final_answer(model_output):
    """
    Extract effects of 'Beauty' from the provided statsmodels results objects.

    Expects model_output to be a dict with keys:
      - 'ols' : baseline OLS results (statsmodels RegressionResultsWrapper)
      - 'fe_instructor' : OLS with C(instructor_id)
      - 'interaction_gender' : OLS with Beauty:is_female

    Returns a dict with keys:
      - "object": nested dict with numeric results (coefficients, se, pvals, 95% CIs,
                  and for the interaction model the female/male marginal effects at Beauty=0)
      - "description": a short human-readable summary interpreting the main findings
                       about the effect of Beauty on Eval.
    """
    import numpy as np

    # t critical: try to use scipy if available, otherwise use normal approx 1.96
    try:
        from scipy import stats
        def t_crit(df): return stats.t.ppf(0.975, df)
    except Exception:
        def t_crit(df): return 1.96

    def summarize_beauty_from_result(res):
        """
        Extract coefficient, se, pval, 95% CI for 'Beauty' and 'Beauty_sq' (if present).
        Returns dict.
        """
        out = {}
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        try:
            ci_df = res.conf_int()
        except Exception:
            ci_df = None

        df_resid = getattr(res, 'df_resid', None)
        tc = t_crit(df_resid) if df_resid is not None else 1.96

        # Helper to safely extract entries
        def safe_get(name):
            if name in params.index:
                coef = float(params[name])
                se = float(bse[name]) if name in bse.index else None
                p = float(pvals[name]) if name in pvals.index else None
                if ci_df is not None and name in ci_df.index:
                    ci_low, ci_high = float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])
                elif se is not None:
                    ci_low, ci_high = coef - tc * se, coef + tc * se
                else:
                    ci_low = ci_high = None
                return {
                    'coef': coef,
                    'se': se,
                    'p_value': p,
                    'ci_95_lower': ci_low,
                    'ci_95_upper': ci_high,
                    'significant_0.05': (p is not None and p < 0.05)
                }
            else:
                return None

        out['Beauty'] = safe_get('Beauty')
        out['Beauty_sq'] = safe_get('Beauty_sq')
        return out

    results = {}
    # 1) OLS baseline
    res_ols = model_output.get('ols')
    if res_ols is None:
        raise ValueError("model_output must contain key 'ols' with a fitted statsmodels result.")
    results['ols'] = summarize_beauty_from_result(res_ols)

    # 2) Instructor fixed effects
    res_fe = model_output.get('fe_instructor')
    if res_fe is None:
        raise ValueError("model_output must contain key 'fe_instructor' with a fitted statsmodels result.")
    results['fe_instructor'] = summarize_beauty_from_result(res_fe)

    # 3) Interaction model: extract Beauty, interaction term, and compute marginal effects for female and male at Beauty=0
    res_int = model_output.get('interaction_gender')
    if res_int is None:
        raise ValueError("model_output must contain key 'interaction_gender' with a fitted statsmodels result.")
    int_summary = summarize_beauty_from_result(res_int)

    # Names: the interaction term in the model formula was 'Beauty:is_female'
    interaction_name = 'Beauty:is_female'
    params = res_int.params
    cov = res_int.cov_params()
    pvals = res_int.pvalues
    bse = res_int.bse
    df_resid = getattr(res_int, 'df_resid', None)
    tc = t_crit(df_resid) if df_resid is not None else 1.96

    # Extract interaction coefficient if present
    if interaction_name in params.index:
        coef_int = float(params[interaction_name])
        se_int = float(bse[interaction_name]) if interaction_name in bse.index else None
        p_int = float(pvals[interaction_name]) if interaction_name in pvals.index else None
        if interaction_name in cov.index and interaction_name in cov.columns:
            # confidence interval via coef +/- t*se
            ci_low_int = float(coef_int - tc * se_int) if se_int is not None else None
            ci_high_int = float(coef_int + tc * se_int) if se_int is not None else None
        else:
            ci_low_int = ci_high_int = None
        int_summary['interaction_Beauty_is_female'] = {
            'coef': coef_int,
            'se': se_int,
            'p_value': p_int,
            'ci_95_lower': ci_low_int,
            'ci_95_upper': ci_high_int,
            'significant_0.05': (p_int is not None and p_int < 0.05)
        }
    else:
        int_summary['interaction_Beauty_is_female'] = None

    # Marginal effects at Beauty = 0 (data's Beauty is centered in original description, so 0 = mean)
    # Male marginal effect = coef_Beauty (+ 2*Beauty_sq*0 -> only coef_Beauty)
    # Female marginal effect = coef_Beauty + coef_interaction (+ 2*Beauty_sq*0)
    male = None
    female = None
    if int_summary.get('Beauty') is not None:
        beta_b = int_summary['Beauty']['coef']
        # male se & p come from parameter 'Beauty' directly
        male_se = int_summary['Beauty']['se']
        male_p = int_summary['Beauty']['p_value']
        male_ci_low = int_summary['Beauty']['ci_95_lower']
        male_ci_high = int_summary['Beauty']['ci_95_upper']
        male = {
            'marginal_at_beauty_0': beta_b,
            'se': male_se,
            'p_value': male_p,
            'ci_95_lower': male_ci_low,
            'ci_95_upper': male_ci_high,
            'significant_0.05': (male_p is not None and male_p < 0.05)
        }

        # female
        if int_summary.get('interaction_Beauty_is_female') is not None and int_summary['interaction_Beauty_is_female'] is not None:
            beta_int = int_summary['interaction_Beauty_is_female']['coef']
            # variance using covariance matrix: Var(beta_b + beta_int) = Var(b) + Var(int) + 2 Cov(b,int)
            var_b = cov.loc['Beauty', 'Beauty'] if ('Beauty' in cov.index and 'Beauty' in cov.columns) else None
            var_int = cov.loc[interaction_name, interaction_name] if (interaction_name in cov.index and interaction_name in cov.columns) else None
            cov_b_int = cov.loc['Beauty', interaction_name] if ('Beauty' in cov.index and interaction_name in cov.columns) else None

            if var_b is not None and var_int is not None and cov_b_int is not None:
                var_female = float(var_b + var_int + 2 * cov_b_int)
                se_female = float(np.sqrt(var_female)) if var_female >= 0 else None
            else:
                # fallback: if covariance info not available, try to approximate using parameter SEs (conservative)
                se_b = int_summary['Beauty']['se']
                se_int = int_summary['interaction_Beauty_is_female']['se']
                if se_b is not None and se_int is not None:
                    se_female = float(np.sqrt(se_b**2 + se_int**2))
                else:
                    se_female = None

            female_coef = float(beta_b + beta_int)
            female_p = None
            female_ci_low = female_ci_high = None
            if se_female is not None:
                # t stat and p-value for female marginal
                t_stat = female_coef / se_female if se_female != 0 else np.nan
                # p-value from t-stat: two-sided
                try:
                    from scipy import stats
                    if df_resid is not None:
                        female_p = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
                    else:
                        female_p = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
                except Exception:
                    # fallback to normal approx
                    from math import erf, sqrt
                    female_p = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
                female_ci_low = float(female_coef - tc * se_female)
                female_ci_high = float(female_coef + tc * se_female)

            female = {
                'marginal_at_beauty_0': female_coef,
                'se': se_female,
                'p_value': female_p,
                'ci_95_lower': female_ci_low,
                'ci_95_upper': female_ci_high,
                'significant_0.05': (female_p is not None and female_p < 0.05)
            }

    int_summary['marginal_effects_at_beauty_0'] = {
        'male': male,
        'female': female
    }

    results['interaction_gender'] = int_summary

    # Build a short human-readable summary description
    def summarize_result_block(name, block):
        if block is None or block.get('Beauty') is None:
            return f"{name}: No 'Beauty' coefficient found."
        b = block['Beauty']['coef']
        p = block['Beauty']['p_value']
        sig = block['Beauty']['significant_0.05']
        sq = block.get('Beauty_sq')
        sq_txt = ""
        if sq is not None and sq.get('coef') is not None:
            sq_coef = sq['coef']
            sq_p = sq['p_value']
            sq_txt = f" The quadratic term (Beauty_sq) has coef={sq_coef:.4f}, p={sq_p:.3g}."
        return (f"{name}: A one-unit increase in Beauty is associated with a change in Eval of "
                f"{b:.4f} (p={p:.3g}){' — statistically significant' if sig else ' — not statistically significant' }."
                + sq_txt)

    desc_lines = [
        "Extracted results for the effect of instructor Beauty on course Eval:",
        summarize_result_block("Baseline OLS", results['ols']),
        summarize_result_block("Instructor fixed effects (C(instructor_id))", results['fe_instructor']),
        # Add interaction interpretation
    ]

    # Add interaction-specific summary
    ig = results['interaction_gender']
    if ig and ig.get('Beauty') is not None:
        b = ig['Beauty']['coef']
        p = ig['Beauty']['p_value']
        inter = ig.get('interaction_Beauty_is_female')
        inter_txt = ""
        if inter is not None:
            ci = inter['coef']
            cip = inter['p_value']
            inter_txt = (f" The interaction term Beauty:is_female has coef={ci:.4f}, p={cip:.3g} "
                         f"({'significant' if inter['significant_0.05'] else 'not significant'}).")
        male_m = ig['marginal_effects_at_beauty_0']['male']
        female_m = ig['marginal_effects_at_beauty_0']['female']
        male_txt = (f" For male instructors (is_female=0) the marginal effect at mean beauty is {male_m['marginal_at_beauty_0']:.4f}"
                    f" (p={male_m['p_value']:.3g})." if male_m is not None else "")
        if female_m is not None and female_m['marginal_at_beauty_0'] is not None:
            female_txt = (f" For female instructors (is_female=1) the marginal effect at mean beauty is {female_m['marginal_at_beauty_0']:.4f}"
                          f" (p={female_m['p_value']:.3g}).")
        else:
            female_txt = ""
        desc_lines.append("Interaction model: " + f"A one-unit increase in Beauty has coef={b:.4f} (p={p:.3g})." + inter_txt + " " + male_txt + " " + female_txt)

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }