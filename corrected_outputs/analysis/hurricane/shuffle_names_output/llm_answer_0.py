def extract_final_answer(model_output):
    """
    Extracts statistics for the Femininity predictor from the provided model_output.
    Expects model_output to be a dict with keys:
      - 'ols_log_deaths': a fitted statsmodels RegressionResultsWrapper (OLS on log-deaths)
      - 'neg_binom_counts': a fitted statsmodels GLMResultsWrapper (NegativeBinomial) or None

    Returns a dict with:
      - "object": a dict containing extracted numeric results for both models and a boolean
                  'support_hypothesis' (True if primary model shows a positive, statistically
                  significant association at alpha=0.05).
      - "description": a short plain-language interpretation of the results in context.
    """
    import numpy as np

    results = {}
    ols = model_output.get('ols_log_deaths')
    nb = model_output.get('neg_binom_counts')

    # Helper to safely extract stats for a given results object and variable name
    def _extract_stats(res, varname='Femininity'):
        if res is None:
            return None
        try:
            coef = float(res.params[varname])
            se = float(res.bse[varname]) if hasattr(res, 'bse') else None
            pval = float(res.pvalues[varname]) if hasattr(res, 'pvalues') else None
            # conf_int may be a DataFrame/ndarray
            try:
                ci = res.conf_int().loc[varname].astype(float)
                ci_lower, ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                # fallback if conf_int returns ndarray with column ordering
                ci_arr = res.conf_int()
                # try to locate variable by index if present
                try:
                    idx = list(res.params.index).index(varname)
                    ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
                except Exception:
                    ci_lower, ci_upper = None, None
            return {
                'coef': coef,
                'se': se,
                'pvalue': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        except Exception:
            return None

    ols_stats = _extract_stats(ols, 'Femininity')
    nb_stats = _extract_stats(nb, 'Femininity') if nb is not None else None

    # Interpretations
    interpretation = {}

    if ols_stats is not None:
        beta = ols_stats['coef']
        # Because outcome is log(TotalDeaths + 1), exponentiating gives multiplicative change
        mult_effect = np.exp(beta)
        percent_change = (mult_effect - 1.0) * 100.0
        ols_stats['multiplicative_effect_on_TotalDeaths_plus1'] = mult_effect
        ols_stats['percent_change_on_TotalDeaths_plus1'] = percent_change
        # Significance flag
        ols_stats['significant'] = (ols_stats['pvalue'] is not None) and (ols_stats['pvalue'] < 0.05)
    else:
        ols_stats = None

    if nb_stats is not None:
        nb_beta = nb_stats['coef']
        nb_mult = np.exp(nb_beta)  # multiplicative effect on expected counts (log link)
        nb_pct = (nb_mult - 1.0) * 100.0
        nb_ci_lower = nb_stats['ci_lower']
        nb_ci_upper = nb_stats['ci_upper']
        # exponentiated CI if available
        try:
            nb_exp_ci_lower = float(np.exp(nb_ci_lower)) if nb_ci_lower is not None else None
            nb_exp_ci_upper = float(np.exp(nb_ci_upper)) if nb_ci_upper is not None else None
        except Exception:
            nb_exp_ci_lower, nb_exp_ci_upper = None, None
        nb_stats['multiplicative_effect_on_counts'] = nb_mult
        nb_stats['percent_change_on_counts'] = nb_pct
        nb_stats['exp_ci_lower'] = nb_exp_ci_lower
        nb_stats['exp_ci_upper'] = nb_exp_ci_upper
        nb_stats['significant'] = (nb_stats['pvalue'] is not None) and (nb_stats['pvalue'] < 0.05)

    # Decide whether results support the hypothesis.
    # Primary criterion: OLS on log-deaths (the stated primary model) should show positive coef and p < 0.05.
    support_hypothesis = False
    support_reason = ""
    if ols_stats is None:
        support_reason = "Primary OLS model results not available."
    else:
        if ols_stats['significant'] and ols_stats['coef'] > 0:
            support_hypothesis = True
            support_reason = ("Primary OLS shows a positive and statistically significant association "
                              "between Femininity and log(TotalDeaths+1).")
        else:
            support_hypothesis = False
            if ols_stats['coef'] > 0:
                support_reason = ("Primary OLS coefficient is positive but not statistically significant "
                                  f"(p = {ols_stats['pvalue']:.3g}).")
            else:
                support_reason = ("Primary OLS coefficient is not positive (coef = "
                                  f"{ols_stats['coef']:.4g}) and does not support the hypothesized direction.")

    # Summarize numeric outputs concisely for the description
    def _fmt(x, digits=3):
        try:
            return f"{x:.{digits}f}"
        except Exception:
            return str(x)

    desc_lines = []
    if ols_stats is not None:
        desc_lines.append(
            "OLS (log(TotalDeaths+1)) -> Femininity coef = "
            f"{_fmt(ols_stats['coef'])}, SE = {_fmt(ols_stats['se'])}, p = {_fmt(ols_stats['pvalue'],3)}; "
            f"CI = [{_fmt(ols_stats['ci_lower'])}, {_fmt(ols_stats['ci_upper'])}]."
        )
        desc_lines.append(
            "Interpreted on the original-count scale: a one-unit increase in Femininity is associated with "
            f"{_fmt(ols_stats['percent_change_on_TotalDeaths_plus1'],3)}% change in (TotalDeaths+1)."
        )
    else:
        desc_lines.append("OLS results for Femininity not available.")

    if nb_stats is not None:
        desc_lines.append(
            "Negative Binomial (counts) -> Femininity coef = "
            f"{_fmt(nb_stats['coef'])}, SE = {_fmt(nb_stats['se'])}, p = {_fmt(nb_stats['pvalue'],3)}; "
            f"exp(coef) = {_fmt(nb_stats['multiplicative_effect_on_counts'],3)} "
            f"(% change = {_fmt(nb_stats['percent_change_on_counts'],3)}%), "
            f"exp(CI) = [{_fmt(nb_stats.get('exp_ci_lower'))}, {_fmt(nb_stats.get('exp_ci_upper'))}]."
        )
    else:
        desc_lines.append("Negative binomial robustness model not available or failed to converge.")

    desc_lines.append("Conclusion: " + ("Supports" if support_hypothesis else "Does not support") +
                      " the hypothesis that more feminine names lead to higher fatalities.")
    desc_lines.append("Reason: " + support_reason)

    description = " ".join(desc_lines)

    output_obj = {
        'ols_femininity': ols_stats,
        'negbin_femininity': nb_stats,
        'support_hypothesis': support_hypothesis,
        'support_reason': support_reason
    }

    return {
        "object": output_obj,
        "description": description
    }