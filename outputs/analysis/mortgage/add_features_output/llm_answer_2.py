def extract_final_answer(model_output):
    """
    Extracts the estimated effect of being female on mortgage approval from the
    model_output produced by the modeling function. Returns a dictionary with:
      - "object": dict with numeric results (AME, SE, z, p, 95% CI, and fallback logit coef info if available)
      - "description": short interpretation in context

    This function is robust to two forms of input:
      - model_output['female_margeff'] is a pandas Series (preferred) with entries
        like 'dy/dx', 'Std. Err.', 'z', 'Pr(>|z|)', 'Conf. Int. Low', 'Cont. Int. Hi'
      - If female_margeff is not available, it will try to extract the logit
        coefficient and standard error from model_output['results_robust'].
    """
    import math
    import re

    result = {
        "object": None,
        "description": None
    }

    fem_margeff = model_output.get('female_margeff', None)
    res = model_output.get('results_robust', None)

    # Helper to compute two-sided p-value from z using normal approximation
    def p_from_z(z):
        # two-sided p-value using erfc: p = erfc(|z|/sqrt(2))
        return float(math.erfc(abs(z) / math.sqrt(2)))

    # If female marginal effect is present, parse it
    if fem_margeff is not None:
        try:
            # Build normalized-index -> value mapping
            norm_dict = {}
            for idx, val in zip(map(str, fem_margeff.index), fem_margeff.values):
                key = re.sub(r'[^a-z0-9]', '', idx.lower())
                norm_dict[key] = float(val) if (val is not None and not (isinstance(val, float) and math.isnan(val))) else None

            def get_val(cands):
                for cand in cands:
                    for k in norm_dict:
                        if cand in k:
                            return norm_dict[k]
                return None

            ame = get_val(['dydx', 'dy', 'ame'])
            se = get_val(['stderr', 'stderror', 'stder', 'std', 'se'])
            z = get_val(['^z$', 'z'])  # 'z' included
            p = get_val(['pr', 'pvalue', 'p'])
            # Confidence interval: look for keys containing 'low' and 'hi' or 'high'
            ci_low = None
            ci_high = None
            for k in norm_dict:
                if 'low' in k:
                    ci_low = norm_dict[k]
                if 'hi' in k or 'high' in k:
                    ci_high = norm_dict[k]

            # If p or z missing, try to compute from ame and se
            if p is None and (z is None) and (ame is not None and se is not None and se != 0):
                z = ame / se
                p = p_from_z(z)

            # Prepare object dict
            obj = {
                'AME': ame,                    # average marginal effect: change in probability
                'AME_std_err': se,
                'AME_z': z,
                'AME_pvalue': p,
                'AME_conf_int_low': ci_low,
                'AME_conf_int_high': ci_high
            }

            # Also attempt to include logit coefficient info if available from results
            if res is not None:
                try:
                    coef = None
                    coef_se = None
                    coef_p = None
                    # results might be statsmodels wrapper or custom; try common attributes
                    if hasattr(res, 'params') and 'female' in res.params.index:
                        coef = float(res.params['female'])
                    if hasattr(res, 'bse') and 'female' in res.bse.index:
                        coef_se = float(res.bse['female'])
                    # try pvalues if present
                    if hasattr(res, 'pvalues') and 'female' in res.pvalues.index:
                        coef_p = float(res.pvalues['female'])
                    # fallback: try cov_params to get se
                    if coef_se is None and hasattr(res, 'cov_params'):
                        try:
                            cov = res.cov_params()
                            if 'female' in cov.index:
                                coef_se = float(math.sqrt(float(cov.loc['female', 'female'])))
                        except Exception:
                            coef_se = None
                    # compute p if missing
                    if coef_p is None and coef is not None and coef_se is not None and coef_se != 0:
                        coef_p = p_from_z(coef / coef_se)
                    if coef is not None:
                        obj['logit_coef'] = coef
                        obj['logit_se'] = coef_se
                        obj['logit_pvalue'] = coef_p
                except Exception:
                    pass

            # Build description
            if obj['AME'] is not None:
                pct = obj['AME'] * 100
                pval = obj['AME_pvalue']
                low = obj['AME_conf_int_low']
                high = obj['AME_conf_int_high']
                if pval is not None:
                    signif = 'statistically significant' if pval < 0.05 else 'not statistically significant'
                    desc = (f"Being female is associated with an average increase in the probability "
                            f"of mortgage approval of {pct:.2f percentage points} (AME = {obj['AME']:.4f}). "
                            f"Two-sided p = {pval:.3f} ({signif}).")
                else:
                    desc = (f"Being female is associated with an average increase in the probability "
                            f"of mortgage approval of {pct:.2f percentage points} (AME = {obj['AME']:.4f}).")
                if low is not None and high is not None:
                    desc += f" 95% CI for the AME: [{low:.4f}, {high:.4f}]."
                result['object'] = obj
                result['description'] = desc
                return result
        except Exception:
            # If anything fails, fall through to try extracting from results_robust
            pass

    # If we get here, try to extract coefficient information from results_robust
    if res is not None:
        try:
            coef = None
            se = None
            pval = None
            ci_lower = None
            ci_upper = None

            # Try many possible attribute names
            if hasattr(res, 'params') and 'female' in getattr(res, 'params').index:
                coef = float(res.params['female'])
            if hasattr(res, 'bse') and 'female' in getattr(res, 'bse').index:
                se = float(res.bse['female'])
            # p-values attribute
            if hasattr(res, 'pvalues') and 'female' in getattr(res, 'pvalues').index:
                pval = float(res.pvalues['female'])
            # cov_params for robust se if available
            if se is None and hasattr(res, 'cov_params'):
                try:
                    cov = res.cov_params()
                    if 'female' in cov.index:
                        se = float(math.sqrt(float(cov.loc['female', 'female'])))
                except Exception:
                    se = None
            # conf_int if available
            if hasattr(res, 'conf_int'):
                try:
                    ci = res.conf_int()
                    if 'female' in ci.index:
                        ci_lower = float(ci.loc['female', 0])
                        ci_upper = float(ci.loc['female', 1])
                except Exception:
                    pass

            # compute p from z if needed
            if pval is None and coef is not None and se is not None and se != 0:
                z = coef / se
                pval = p_from_z(z)

            obj = {
                'logit_coef': coef,
                'logit_se': se,
                'logit_pvalue': pval,
                'logit_conf_int_low': ci_lower,
                'logit_conf_int_high': ci_upper
            }
            # Interpretation: convert logit coef to sign statement
            if coef is not None:
                direction = 'increase' if coef > 0 else 'decrease' if coef < 0 else 'no change'
                desc = (f"The log-odds coefficient on female is {coef:.4f}, which implies a {direction} "
                        f"in the odds of approval for female applicants. ")
                if pval is not None:
                    signif = 'statistically significant' if pval < 0.05 else 'not statistically significant'
                    desc += f"P-value = {pval:.3f} ({signif})."
                else:
                    desc += "P-value not available."
                # Note: without marginal effect, we do not translate to probability points
                result['object'] = obj
                result['description'] = desc
                return result
        except Exception:
            pass

    # If nothing could be extracted:
    result['object'] = None
    result['description'] = ("Could not extract the female effect from the provided model_output. "
                             "Ensure model_output contains 'female_margeff' (pandas Series) or a "
                             "results object with params/bse/pvalues.")
    return result