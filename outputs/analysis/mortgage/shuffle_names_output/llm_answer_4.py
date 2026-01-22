def extract_final_answer(model_output):
    """
    Extracts the estimated effect of being female on mortgage approval from the model output.

    Returns a dictionary with keys:
      - "object": a dict with numeric results (AME if available, else log-odds coef and odds ratio),
      - "description": a short plain-language interpretation.

    The function attempts to read the average marginal effect from model_output['marginal_effects']
    first. If that's not available, it falls back to model_output['logit_results'] (coefficients).
    """
    import math
    from math import exp
    try:
        from scipy.stats import norm
    except Exception:
        norm = None

    me = model_output.get('marginal_effects', None)
    res = model_output.get('logit_results', None)

    # Helper to compute p-value from effect and se if needed
    def p_from_effect_se(effect, se):
        if se is None or se == 0 or effect is None:
            return None
        z = effect / se
        if norm is not None:
            return float(2 * (1 - norm.cdf(abs(z))))
        # fallback using erf
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return float(p)

    # Try to extract from marginal effects table (preferred)
    if me is not None:
        try:
            # find the Female row (case-insensitive)
            female_idx = None
            for idx in me.index:
                if str(idx).strip().lower() == 'female':
                    female_idx = idx
                    break
            if female_idx is None:
                # try substring match
                for idx in me.index:
                    if 'female' in str(idx).strip().lower():
                        female_idx = idx
                        break

            if female_idx is not None:
                row = me.loc[female_idx]

                # map columns by lowercased name for robust lookup
                colmap = {c.lower(): c for c in me.columns}

                # find dy/dx column
                dy_col = None
                for k in colmap:
                    if 'dy' in k and ('dx' in k or '/' in k):
                        dy_col = colmap[k]; break
                if dy_col is None:
                    for k in colmap:
                        if 'marg' in k or 'effect' in k or 'ame' in k:
                            dy_col = colmap[k]; break

                std_col = None
                for k in colmap:
                    if 'std' in k:
                        std_col = colmap[k]; break

                # confidence interval low/high
                ci_low_col = None
                ci_high_col = None
                for k in colmap:
                    if ('conf' in k and 'low' in k) or ('0.025' in k) or ('low' in k and 'int' in k):
                        ci_low_col = colmap[k]; break
                for k in colmap:
                    if ('conf' in k and ('hi' in k or 'high' in k)) or ('0.975' in k) or ('high' in k and 'int' in k):
                        ci_high_col = colmap[k]; break
                # fallback find any column with 'low'/'high' or '0.025'/'0.975'
                if ci_low_col is None:
                    for k in colmap:
                        if 'low' in k and 'conf' not in k:
                            ci_low_col = colmap[k]; break
                if ci_high_col is None:
                    for k in colmap:
                        if ('hi' in k or 'high' in k) and 'conf' not in k:
                            ci_high_col = colmap[k]; break

                # p-value column
                p_col = None
                for k in colmap:
                    if k.startswith('p') or 'p>' in k or 'p|' in k or 'pvalue' in k or 'p-value' in k:
                        p_col = colmap[k]; break

                # Extract numeric values if present
                def safe_get(series, col):
                    try:
                        return None if col is None else float(series[col])
                    except Exception:
                        return None

                ame = safe_get(row, dy_col)
                se = safe_get(row, std_col)
                ci_low = safe_get(row, ci_low_col)
                ci_high = safe_get(row, ci_high_col)
                pval = safe_get(row, p_col)
                if pval is None:
                    pval = p_from_effect_se(ame, se)

                result_obj = {
                    'effect_type': 'average_marginal_effect',
                    'AME': ame,
                    'std_err': se,
                    '95%_CI_low': ci_low,
                    '95%_CI_high': ci_high,
                    'p_value': pval
                }

                # Construct a short interpretation
                if ame is None:
                    descr = "Female effect found in marginal effects table but numeric columns could not be read."
                else:
                    # interpret sign and significance
                    sig = (pval is not None and pval < 0.05)
                    direction = "increase" if ame > 0 else ("decrease" if ame < 0 else "no change")
                    percent_fmt = f"{ame*100:.2f} percentage points" if abs(ame) < 1.0 else f"{ame:.4f} (in probability units)"
                    descr = (
                        f"Being female is associated with a {direction} in the probability of mortgage approval "
                        f"of about {percent_fmt} (AME = {ame:.4f})."
                    )
                    if (result_obj['95%_CI_low'] is not None) and (result_obj['95%_CI_high'] is not None):
                        descr += f" 95% CI [{result_obj['95%_CI_low']:.4f}, {result_obj['95%_CI_high']:.4f}]."
                    if pval is not None:
                        descr += f" p = {pval:.3f}."
                    descr += " " + ("This effect is statistically significant at the 5% level." if sig else "This effect is not statistically significant at the 5% level.")
                return {"object": result_obj, "description": descr}

        except Exception as e:
            # proceed to fallback
            pass

    # Fallback to coefficient from logit_results (log-odds)
    if res is not None:
        try:
            # try to get female coef, bse, pval from results object
            params = getattr(res, 'params', None)
            bse = getattr(res, 'bse', None)
            pvals = getattr(res, 'pvalues', None)
            cov = None
            try:
                cov = res.cov_params()
            except Exception:
                cov = None

            coef = None
            se = None
            pval = None
            if params is not None and 'Female' in params.index:
                coef = float(params['Female'])
            else:
                # try lowercase match
                if params is not None:
                    for idx in params.index:
                        if str(idx).strip().lower() == 'female':
                            coef = float(params[idx]); break
            if bse is not None and 'Female' in bse.index:
                se = float(bse['Female'])
            else:
                if bse is not None:
                    for idx in bse.index:
                        if str(idx).strip().lower() == 'female':
                            se = float(bse[idx]); break
            if pvals is not None and 'Female' in pvals.index:
                pval = float(pvals['Female'])
            else:
                if pvals is not None:
                    for idx in pvals.index:
                        if str(idx).strip().lower() == 'female':
                            pval = float(pvals[idx]); break

            if pval is None:
                pval = p_from_effect_se(coef, se)

            # confidence interval for coef
            ci_low = ci_high = None
            if se is not None and coef is not None:
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

            # odds ratio and its CI
            orr = None
            or_ci_low = or_ci_high = None
            if coef is not None:
                orr = float(math.exp(coef))
                if ci_low is not None and ci_high is not None:
                    or_ci_low = float(math.exp(ci_low))
                    or_ci_high = float(math.exp(ci_high))

            result_obj = {
                'effect_type': 'logit_coefficient',
                'coef_log_odds': coef,
                'std_err': se,
                '95%_CI_log_odds_low': ci_low,
                '95%_CI_log_odds_high': ci_high,
                'odds_ratio': orr,
                '95%_CI_or_low': or_ci_low,
                '95%_CI_or_high': or_ci_high,
                'p_value': pval
            }

            # Interpretation
            if coef is None:
                descr = "Could not find a 'Female' coefficient in the provided logit results."
            else:
                direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no different")
                descr = (
                    f"In log-odds, being female is associated with {direction} log-odds of mortgage approval "
                    f"(coef = {coef:.4f})."
                )
                if orr is not None:
                    descr += f" This corresponds to an odds ratio of {orr:.3f}."
                if (result_obj['95%_CI_log_odds_low'] is not None) and (result_obj['95%_CI_log_odds_high'] is not None):
                    descr += f" 95% CI for coef [{ci_low:.4f}, {ci_high:.4f}]."
                if pval is not None:
                    descr += f" p = {pval:.3f}."
                sig = (pval is not None and pval < 0.05)
                descr += " " + ("Statistically significant at the 5% level." if sig else "Not statistically significant at the 5% level.")
            return {"object": result_obj, "description": descr}

        except Exception:
            pass

    # If we get here, nothing could be read
    return {
        "object": None,
        "description": "Unable to extract the female effect from the provided model_output. Ensure 'marginal_effects' or 'logit_results' with a 'Female' parameter are present."
    }