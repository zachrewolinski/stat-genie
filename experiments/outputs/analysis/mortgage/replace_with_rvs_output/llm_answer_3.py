import numpy as np


def extract_final_answer(model_output):
    """
    Extracts the estimated effect of applicant gender (female) on mortgage acceptance
    from the model_output produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing numeric results for 'female' (odds ratio, 95% CI, p-value,
                  coefficient on log-odds scale if available, robust se if available, and a
                  boolean 'significant' at alpha=0.05).
      - "description": a short plain-language interpretation of the result in context.

    The function is defensive: it first tries to read a 'summary_table' DataFrame (preferred),
    then falls back to 'model_results_robust' attributes if needed.
    """
    # Prepare default return structure
    result_obj = {
        'odds_ratio': None,
        'ci_95': [None, None],
        'pvalue': None,
        'coef_log_odds': None,
        'coef_ci_95': [None, None],
        'robust_se': None,
        'significant': None
    }

    # Try summary_table first (expected structure)
    summary = model_output.get('summary_table') if isinstance(model_output, dict) else None
    if summary is not None and 'female' in getattr(summary, "index", []):
        try:
            row = summary.loc['female']
            or_val = float(row.get('OR', np.nan))
            ci_low = float(row.get('2.5%', np.nan))
            ci_high = float(row.get('97.5%', np.nan))
            pval = float(row.get('pvalue', np.nan))

            result_obj['odds_ratio'] = or_val
            result_obj['ci_95'] = [ci_low, ci_high]
            result_obj['pvalue'] = pval
            result_obj['significant'] = (pval < 0.05) if (pval is not None and not np.isnan(pval)) else None
        except Exception:
            # If anything goes wrong reading the table, leave these as None and try fallback below
            pass

    # If any key information is still missing, try model_results_robust fallback
    robust = model_output.get('model_results_robust') if isinstance(model_output, dict) else None
    if (result_obj['odds_ratio'] is None or result_obj['pvalue'] is None) and robust is not None:
        try:
            # coefficient on log-odds scale
            if hasattr(robust, 'params') and 'female' in getattr(robust.params, "index", []):
                coef = float(robust.params['female'])
                result_obj['coef_log_odds'] = coef
                # odds ratio from coef if not already set
                if result_obj['odds_ratio'] is None:
                    result_obj['odds_ratio'] = float(np.exp(coef))
                # robust se if available
                if hasattr(robust, 'bse') and 'female' in getattr(robust.bse, "index", []):
                    se = float(robust.bse['female'])
                    result_obj['robust_se'] = se
                    z = coef / se if se != 0 else None
                else:
                    se = None
                    z = None
                # p-value
                if hasattr(robust, 'pvalues') and 'female' in getattr(robust.pvalues, "index", []):
                    pval = float(robust.pvalues['female'])
                    result_obj['pvalue'] = pval
                else:
                    # compute approx p from z if possible
                    if z is not None:
                        try:
                            from scipy import stats
                            pval = 2 * (1 - stats.norm.cdf(abs(z)))
                            result_obj['pvalue'] = float(pval)
                        except Exception:
                            pass
                # confidence interval on coefficient if conf_int available
                if hasattr(robust, 'conf_int'):
                    try:
                        ci_obj = robust.conf_int() if callable(robust.conf_int) else robust.conf_int
                        if 'female' in getattr(ci_obj, "index", []):
                            lower, upper = float(ci_obj.loc['female', 0]), float(ci_obj.loc['female', 1])
                            result_obj['coef_ci_95'] = [lower, upper]
                            # convert to OR scale if not present
                            if result_obj['ci_95'] == [None, None]:
                                result_obj['ci_95'] = [float(np.exp(lower)), float(np.exp(upper))]
                    except Exception:
                        pass
                # significance flag if pvalue present
                if result_obj['pvalue'] is not None:
                    result_obj['significant'] = (result_obj['pvalue'] < 0.05)
        except Exception:
            pass

    # Build a plain-language description
    desc_parts = []
    if result_obj['odds_ratio'] is not None:
        desc_parts.append(
            f"Estimated odds ratio for female vs male = {result_obj['odds_ratio']:.3f}."
        )
        if result_obj['ci_95'][0] is not None and result_obj['ci_95'][1] is not None:
            desc_parts.append(
                f"95% CI = [{result_obj['ci_95'][0]:.3f}, {result_obj['ci_95'][1]:.3f}]."
            )
    elif result_obj['coef_log_odds'] is not None:
        desc_parts.append(
            f"Estimated log-odds coefficient for female = {result_obj['coef_log_odds']:.3f}."
        )
        if result_obj['coef_ci_95'][0] is not None and result_obj['coef_ci_95'][1] is not None:
            desc_parts.append(
                f"95% CI (log-odds) = [{result_obj['coef_ci_95'][0]:.3f}, {result_obj['coef_ci_95'][1]:.3f}]."
            )

    if result_obj['pvalue'] is not None:
        desc_parts.append(f"p-value = {result_obj['pvalue']:.3f}.")
        if result_obj['significant'] is True:
            desc_parts.append("This is statistically significant at the 0.05 level.")
        elif result_obj['significant'] is False:
            desc_parts.append("This is NOT statistically significant at the 0.05 level.")
    else:
        desc_parts.append("p-value not available; cannot assess statistical significance.")

    # Final interpretation in context
    # If OR available, translate magnitude
    if result_obj['odds_ratio'] is not None:
        or_pct = (result_obj['odds_ratio'] - 1) * 100
        sign = "higher" if or_pct > 0 else ("lower" if or_pct < 0 else "the same")
        desc_parts.append(
            f"Interpretation: female applicants have an estimated {abs(or_pct):.1f}% {sign} odds of mortgage acceptance "
            f"relative to male applicants, after adjusting for the listed controls."
        )
        if result_obj['significant'] is False:
            desc_parts.append(
                "Because the confidence interval includes 1 (and p > 0.05), there is no statistically reliable evidence of a gender effect."
            )
    else:
        desc_parts.append(
            "Could not compute/locate an odds ratio for female; see numeric outputs for details."
        )

    description = " ".join(desc_parts)

    return {
        "object": result_obj,
        "description": description
    }