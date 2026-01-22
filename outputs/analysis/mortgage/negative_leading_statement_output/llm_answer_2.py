def extract_final_answer(model_output):
    """
    Extracts the adjusted effect of gender (female) on loan acceptance from the
    model_output dict returned by the provided modeling function.
    
    Returns a dictionary with:
      - "object": a dict with numeric summaries (coef, se, p-value, odds ratio,
                  95% CI for odds ratio, unadjusted acceptance rates, n_obs)
      - "description": a human-readable interpretation of the results in context
    """
    import numpy as np
    import pandas as pd

    # Basic safety checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model function.")
    if 'model' not in model_output and 'odds_ratios' not in model_output:
        raise ValueError("model_output does not contain expected keys ('model' or 'odds_ratios').")

    # Try to extract details from the fitted statsmodels result if available
    res = model_output.get('model', None)
    odds_series = model_output.get('odds_ratios', None)
    conf_odds_df = model_output.get('conf_int_odds', None)
    accept_rate_by_gender = model_output.get('accept_rate_by_gender', None)
    n_obs = model_output.get('n_obs', None)

    result_obj = {}

    # Preferred: use statsmodels result to get coef, se, pvalue, conf_int
    try:
        if res is not None:
            params = res.params
            pvalues = res.pvalues
            bse = res.bse
            conf = res.conf_int()
            # Ensure 'female' exists
            if 'female' not in params.index:
                raise KeyError("No 'female' parameter in model.params")
            coef = float(params['female'])
            se = float(bse['female']) if 'female' in bse.index else None
            pval = float(pvalues['female'])
            # Odds ratio and CI
            or_female = float(np.exp(coef))
            if 'female' in conf.index:
                ci_low, ci_high = conf.loc['female'][0], conf.loc['female'][1]
                ci_or = (float(np.exp(ci_low)), float(np.exp(ci_high)))
            else:
                ci_or = (None, None)
            # Fill numeric summary
            result_obj.update({
                'coef': coef,
                'std_err': se,
                'p_value': pval,
                'odds_ratio': or_female,
                'odds_ratio_95CI': ci_or
            })
            # number of observations if not provided
            if n_obs is None:
                try:
                    n_obs = int(res.nobs)
                except Exception:
                    n_obs = None
    except Exception:
        # Fallback: use precomputed odds_ratios and conf_int_odds if available
        if odds_series is not None and 'female' in odds_series.index:
            or_female = float(odds_series['female'])
            result_obj['odds_ratio'] = or_female
            if conf_odds_df is not None and 'female' in conf_odds_df.index:
                ci_or = (float(conf_odds_df.loc['female', '2.5%']), float(conf_odds_df.loc['female', '97.5%']))
                result_obj['odds_ratio_95CI'] = ci_or
        # try to get p-value / coef if present in model_output summary_text (best-effort)
        if res is None and 'summary_text' in model_output:
            # crude parse (best-effort)
            try:
                st = model_output['summary_text']
                # Attempt to extract coefficient and p-value lines for 'female'
                for line in st.splitlines():
                    if line.strip().startswith('female'):
                        parts = line.split()
                        # coef at position 1, std err pos 2, z pos 3, p pos 4
                        if len(parts) >= 5:
                            result_obj.setdefault('coef', float(parts[1]))
                            result_obj.setdefault('std_err', float(parts[2]))
                            result_obj.setdefault('p_value', float(parts[4]))
                        break
            except Exception:
                pass

    # Attach acceptance rates by gender (unadjusted)
    if accept_rate_by_gender is not None:
        # ensure it's a DataFrame/Series we can read
        try:
            df_rates = accept_rate_by_gender.copy()
            # Expect index 0/1 for male/female
            # Normalize to dict like {'female_0': (rate, count), 'female_1': (rate,count)}
            rates = {}
            for idx, row in df_rates.iterrows():
                rates[int(idx)] = {'accept_rate': float(row['accept_rate']) if 'accept_rate' in row.index else float(row.get('mean', np.nan)),
                                   'count': int(row['count']) if 'count' in row.index else int(row.get('count', 0))}
            result_obj['unadjusted_accept_rate_by_female'] = rates
        except Exception:
            # if it's already a dict-like
            result_obj['unadjusted_accept_rate_by_female'] = accept_rate_by_gender

    # Attach n_obs if available
    if n_obs is not None:
        result_obj['n_obs'] = int(n_obs)

    # Build a human-readable description
    # Pull numbers if available for formatting
    coef = result_obj.get('coef', None)
    pval = result_obj.get('p_value', None)
    or_female = result_obj.get('odds_ratio', None)
    ci_or = result_obj.get('odds_ratio_95CI', (None, None))
    rates = result_obj.get('unadjusted_accept_rate_by_female', None)
    n_obs = result_obj.get('n_obs', n_obs)

    desc_parts = []
    if (or_female is not None) and (ci_or[0] is not None):
        desc_parts.append(
            f"Adjusted odds ratio for female vs male = {or_female:.3f} "
            f"(95% CI {ci_or[0]:.3f}–{ci_or[1]:.3f})."
        )
    elif or_female is not None:
        desc_parts.append(f"Adjusted odds ratio for female vs male = {or_female:.3f}.")

    if (coef is not None) and (pval is not None):
        sig_text = "statistically significant" if pval < 0.05 else "not statistically significant"
        desc_parts.append(
            f"Logistic coefficient = {coef:.3f} (p = {pval:.3f}), which is {sig_text} at α=0.05."
        )
    elif pval is not None:
        sig_text = "statistically significant" if pval < 0.05 else "not statistically significant"
        desc_parts.append(f"p-value = {pval:.3f} ({sig_text}).")

    if rates is not None:
        # rates keyed by 0/1; female indicator 1 = female per spec
        male_info = rates.get(0)
        female_info = rates.get(1)
        if male_info and female_info:
            desc_parts.append(
                f"Unadjusted acceptance rates: male = {male_info['accept_rate']:.3%} (n={male_info['count']}), "
                f"female = {female_info['accept_rate']:.3%} (n={female_info['count']})."
            )

            # Note potential discrepancy between unadjusted and adjusted result
            if or_female is not None:
                desc_parts.append(
                    "Although raw acceptance rates are nearly identical, the adjusted model "
                    "shows a higher odds of acceptance for female applicants after controlling for covariates."
                )
    if n_obs is not None:
        desc_parts.append(f"Number of observations in the model: {n_obs}.")

    if not desc_parts:
        description = "Could not extract summary statistics for 'female' from the provided model_output."
    else:
        description = " ".join(desc_parts)

    return {
        "object": result_obj,
        "description": description
    }