def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of applicant gender (female) on mortgage approval
    from the modeling output and return a concise interpretable result.

    Returns a dictionary with:
      - "object": dict containing numeric results (logit coef, p-value, odds ratio and 95% CI,
                  average marginal effect and its p-value, number of observations when available)
      - "description": short plain-language interpretation answering whether gender affects approval.

    The function accepts either:
      - the dictionary returned by the provided model() function (contains 'statsmodels_result_object',
        'odds_ratio_female', 'odds_ratio_female_ci_lower', 'odds_ratio_female_ci_upper',
        and 'marginal_effects_text'), or
      - directly a statsmodels BinaryResultsWrapper object.
    """
    import math
    import re
    import numpy as np
    import pandas as pd

    # Resolve statsmodels result object and any textual outputs if present
    res = None
    marg_text = None
    top_level = {}
    if isinstance(model_output, dict):
        res = model_output.get('statsmodels_result_object', None)
        marg_text = model_output.get('marginal_effects_text', None)
        top_level = model_output
    else:
        # assume it's a statsmodels results object
        res = model_output

    if res is None:
        raise ValueError("Could not find a statsmodels results object in model_output.")

    results = {
        'coef_logit': None,
        'coef_p_value': None,
        'odds_ratio': None,
        'odds_ratio_ci': (None, None),
        'marginal_effect': None,
        'marginal_effect_p_value': None,
        'n_obs': None
    }

    # Extract coefficient, p-value, CI, obs from the statsmodels result if possible
    try:
        if 'female' in res.params.index:
            coef = float(res.params['female'])
            pval = float(res.pvalues['female'])
            ci_low_logit, ci_high_logit = res.conf_int().loc['female'].astype(float)
            # convert to odds ratio scale
            or_val = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low_logit))
            or_ci_high = float(np.exp(ci_high_logit))

            results['coef_logit'] = coef
            results['coef_p_value'] = pval
            results['odds_ratio'] = or_val
            results['odds_ratio_ci'] = (or_ci_low, or_ci_high)
        else:
            # female not in model params
            pass
    except Exception:
        # best-effort fallback to top-level odds ratio keys if present
        try:
            or_val = top_level.get('odds_ratio_female', None)
            or_ci_low = top_level.get('odds_ratio_female_ci_lower', None)
            or_ci_high = top_level.get('odds_ratio_female_ci_upper', None)
            if or_val is not None:
                results['odds_ratio'] = float(or_val)
                results['odds_ratio_ci'] = (float(or_ci_low), float(or_ci_high))
        except Exception:
            pass

    # Number of observations if available
    try:
        if hasattr(res, 'nobs'):
            results['n_obs'] = int(res.nobs)
        else:
            # try parse from model summary text if present
            ms = top_level.get('model_summary', None)
            if ms:
                m = re.search(r'No\. Observations:\s*([0-9,]+)', ms)
                if m:
                    results['n_obs'] = int(m.group(1).replace(',', ''))
    except Exception:
        results['n_obs'] = results.get('n_obs', None)

    # Extract marginal effect and its p-value.
    # Preferred: compute from statsmodels margeff if available; fallback: parse marginal_effects_text.
    me_value = None
    me_pval = None
    try:
        # try compute directly
        marg = res.get_margeff(at='overall', method='dydx')
        # try access summary_frame() if available
        if hasattr(marg, 'summary_frame'):
            me_df = marg.summary_frame()
            # find a row named 'female' (or index containing 'female')
            female_idx = None
            for idx in me_df.index:
                if str(idx).strip() == 'female':
                    female_idx = idx
                    break
            if female_idx is None:
                # try case-insensitive match
                for idx in me_df.index:
                    if 'female' in str(idx).lower():
                        female_idx = idx
                        break
            if female_idx is not None:
                # column names may vary; try common possibilities
                cols = list(me_df.columns)
                # find dy/dx column
                dycol = None
                pcol = None
                for c in cols:
                    lc = str(c).lower()
                    if 'dy/dx' in lc or 'dy/dx' == lc or 'dy/dx' in str(c):
                        dycol = c
                    if 'p' in lc and ('|' in lc or 'p>' in lc or 'p-value' in lc or 'p' == lc or 'p>|z|' in lc):
                        pcol = c
                # fallback to positional columns
                if dycol is None:
                    dycol = cols[0]  # often dy/dx is first numeric column
                if pcol is None and len(cols) >= 4:
                    pcol = cols[3]  # often P>|z| is 4th column
                me_value = float(me_df.loc[female_idx, dycol])
                me_pval = float(me_df.loc[female_idx, pcol])
        else:
            raise Exception("marginal result has no summary_frame()")
    except Exception:
        # fallback: parse textual marginal effects (if present)
        if marg_text is None and isinstance(top_level, dict):
            marg_text = top_level.get('marginal_effects_text', None)
        if marg_text:
            # find the line that starts with 'female'
            for line in marg_text.splitlines():
                if line.strip().startswith('female'):
                    tokens = re.split(r'\s+', line.strip())
                    # Expect tokens: [variable, dy/dx, std err, z, P>|z|, [0.025, 0.975]]
                    if len(tokens) >= 5:
                        try:
                            me_value = float(tokens[1])
                        except Exception:
                            me_value = None
                        try:
                            me_pval = float(tokens[4])
                        except Exception:
                            me_pval = None
                    break

    # Finalize marginal effect results
    if me_value is not None:
        results['marginal_effect'] = me_value
    if me_pval is not None:
        results['marginal_effect_p_value'] = me_pval

    # Build plain-language description / conclusion
    desc_parts = []
    if results['odds_ratio'] is not None:
        or_val = results['odds_ratio']
        or_lo, or_hi = results['odds_ratio_ci']
        coef_p = results['coef_p_value']
        desc_parts.append(
            f"Estimated odds ratio for female vs. male = {or_val:.3f} "
            f"(95% CI: {or_lo:.3f} to {or_hi:.3f})."
        )
        if coef_p is not None:
            sig = "statistically significant (p = {:.3f})".format(coef_p) if coef_p < 0.05 else "not statistically significant (p = {:.3f})".format(coef_p)
            desc_parts.append(f"This coefficient is {sig}.")
    else:
        desc_parts.append("Odds ratio for female not available from the provided output.")

    if results['marginal_effect'] is not None:
        me = results['marginal_effect']
        me_p = results['marginal_effect_p_value']
        # convert to percentage points for readability
        desc_parts.append(
            f"Average marginal effect of being female on probability of acceptance = {me:.4f} "
            f"(≈ {me*100:.2f} percentage points)."
        )
        if me_p is not None:
            sig = "statistically significant" if me_p < 0.05 else "not statistically significant"
            desc_parts.append(f"Marginal effect is {sig} (p = {me_p:.3f}).")
    else:
        desc_parts.append("Average marginal effect for female not available from the provided output.")

    if results['n_obs'] is not None:
        desc_parts.append(f"Number of observations: {results['n_obs']}.")

    # Short direct answer to the task question
    # Based on extracted statistics, determine the yes/no conclusion at alpha=0.05
    conclusion = "Unable to determine effect of gender from the provided output."
    try:
        if results['coef_p_value'] is not None:
            if results['coef_p_value'] < 0.05:
                # direction from odds ratio / coef
                if results['odds_ratio'] is not None and results['odds_ratio'] > 1:
                    conclusion = "Yes — being female is associated with higher odds of mortgage approval (statistically significant)."
                elif results['odds_ratio'] is not None and results['odds_ratio'] < 1:
                    conclusion = "Yes — being female is associated with lower odds of mortgage approval (statistically significant)."
                else:
                    # fallback to coef sign
                    if results['coef_logit'] is not None:
                        if results['coef_logit'] > 0:
                            conclusion = "Yes — being female is associated with higher odds of mortgage approval (statistically significant)."
                        else:
                            conclusion = "Yes — being female is associated with lower odds of mortgage approval (statistically significant)."
            else:
                conclusion = "No — the effect of being female on approval is not statistically significant at the 5% level."
    except Exception:
        pass

    description = " ".join(desc_parts) + " " + conclusion

    return {
        "object": results,
        "description": description
    }