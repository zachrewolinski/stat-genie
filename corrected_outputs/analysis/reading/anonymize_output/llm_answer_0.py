def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    computes the marginal effect of ReaderView for dyslexic and non-dyslexic readers
    from a statsmodels RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": nested dict containing numeric results
      - "description": human-readable interpretation and final yes/no conclusion
                      to the question "Does Reader View improve reading speed for
                      individuals with dyslexia?"
    """
    import numpy as np
    from scipy import stats

    res = model_output  # expected statsmodels RegressionResultsWrapper

    # helper: get params, cov, conf_int, pvalues
    params = res.params
    cov = res.cov_params()
    pvalues = res.pvalues
    conf = res.conf_int(alpha=0.05)
    df_resid = getattr(res, "df_resid", None)
    if df_resid is None:
        # fallback to large-sample normal approx
        df_resid = np.inf

    # Identify parameter names robustly
    param_names = list(params.index)

    # Find main ReaderView param (contains 'ReaderView' but no ':' indicating interaction)
    reader_main = None
    for n in param_names:
        if 'ReaderView' in n and ':' not in n:
            reader_main = n
            break

    # Find interaction term containing both ReaderView and Dyslexia (contains ':')
    interaction_name = None
    for n in param_names:
        if 'ReaderView' in n and 'Dyslexia' in n and ':' in n:
            interaction_name = n
            break

    # Find Dyslexia main (for completeness)
    dys_main = None
    for n in param_names:
        if 'Dyslexia' in n and ':' not in n:
            dys_main = n
            break

    if reader_main is None:
        return {
            "object": None,
            "description": "Could not find a main effect parameter for 'ReaderView' in model parameters. "
                           "Parameter names present: " + ", ".join(param_names)
        }

    # Extract main effect stats
    b_reader = float(params[reader_main])
    se_reader = float(np.sqrt(cov.loc[reader_main, reader_main]))
    p_reader = float(pvalues[reader_main]) if reader_main in pvalues.index else None
    ci_reader = (float(conf.loc[reader_main, 0]), float(conf.loc[reader_main, 1]))

    # Interaction stats (if present)
    if interaction_name is not None:
        b_inter = float(params[interaction_name])
        se_inter = float(np.sqrt(cov.loc[interaction_name, interaction_name]))
        p_inter = float(pvalues[interaction_name]) if interaction_name in pvalues.index else None
        ci_inter = (float(conf.loc[interaction_name, 0]), float(conf.loc[interaction_name, 1]))
    else:
        b_inter = 0.0
        se_inter = None
        p_inter = None
        ci_inter = (None, None)

    # Marginal effect of ReaderView for non-dyslexic (Dyslexia=0) is b_reader
    eff_non = b_reader
    # SE, t, p, CI for non-dyslexic effect
    se_non = se_reader
    t_non = eff_non / se_non if se_non is not None and se_non > 0 else np.nan
    if np.isfinite(df_resid):
        p_non = float(2 * stats.t.sf(abs(t_non), df_resid))
        t_crit = stats.t.ppf(0.975, df_resid)
    else:
        p_non = float(2 * (1 - stats.norm.cdf(abs(t_non))))
        t_crit = stats.norm.ppf(0.975)
    ci_non = (eff_non - t_crit * se_non, eff_non + t_crit * se_non)

    # Marginal effect of ReaderView for dyslexic (Dyslexia=1) is b_reader + b_inter
    eff_dys = b_reader + b_inter
    # Compute SE of sum: var(b_reader) + var(b_inter) + 2 cov(b_reader, b_inter)
    if interaction_name is not None and interaction_name in cov.index:
        cov_b = cov.loc[reader_main, interaction_name]
        var_sum = cov.loc[reader_main, reader_main] + cov.loc[interaction_name, interaction_name] + 2 * cov_b
        se_dys = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)
        t_dys = eff_dys / se_dys if se_dys is not None and se_dys > 0 else np.nan
        if np.isfinite(df_resid):
            p_dys = float(2 * stats.t.sf(abs(t_dys), df_resid))
        else:
            p_dys = float(2 * (1 - stats.norm.cdf(abs(t_dys))))
        ci_dys = (eff_dys - t_crit * se_dys, eff_dys + t_crit * se_dys)
    else:
        # No interaction present -> effect for dyslexic equals main effect
        se_dys = se_non
        t_dys = t_non
        p_dys = p_non
        ci_dys = ci_non

    # Convert log-coefficient to percent multiplicative change in wpm:
    # percent_change = 100 * (exp(beta) - 1)
    pct_non = 100.0 * (np.exp(eff_non) - 1.0)
    pct_non_ci = (100.0 * (np.exp(ci_non[0]) - 1.0), 100.0 * (np.exp(ci_non[1]) - 1.0))

    pct_dys = 100.0 * (np.exp(eff_dys) - 1.0)
    pct_dys_ci = (100.0 * (np.exp(ci_dys[0]) - 1.0), 100.0 * (np.exp(ci_dys[1]) - 1.0))

    # Build output object
    result_obj = {
        "parameter_names": {
            "reader_main": reader_main,
            "interaction": interaction_name,
            "dyslexia_main": dys_main
        },
        "reader_main": {
            "coef": b_reader,
            "se": se_reader,
            "p_value": p_reader,
            "ci95": ci_reader,
            "pct_change_wpm": pct_non,
            "pct_change_wpm_ci95": pct_non_ci
        },
        "interaction": {
            "name": interaction_name,
            "coef": b_inter,
            "se": se_inter,
            "p_value": p_inter,
            "ci95": ci_inter
        },
        "effect_non_dyslexic": {
            "coef_log_wpm": eff_non,
            "se": se_non,
            "t": t_non,
            "p_value": p_non,
            "ci95_log_wpm": ci_non,
            "pct_change_wpm": pct_non,
            "pct_change_wpm_ci95": pct_non_ci
        },
        "effect_dyslexic": {
            "coef_log_wpm": eff_dys,
            "se": se_dys,
            "t": t_dys,
            "p_value": p_dys,
            "ci95_log_wpm": ci_dys,
            "pct_change_wpm": pct_dys,
            "pct_change_wpm_ci95": pct_dys_ci
        },
        "raw_params": params.to_dict()
    }

    # Interpretation & final answer (yes/no) for the question:
    # "Does Reader View improve reading speed for individuals with dyslexia?"
    # We define "improve" as a positive effect on log_wpm (i.e., pct change > 0)
    # that is statistically significant at alpha = 0.05 for the dyslexic marginal effect.
    final_conclusion = {}
    if np.isnan(eff_dys) or p_dys is None:
        final_answer = "inconclusive"
        comment = ("Could not compute the marginal effect for dyslexic readers (interaction term missing or "
                   "covariance not available). Inspect model parameters manually.")
    else:
        if (eff_dys > 0) and (p_dys < 0.05):
            final_answer = "yes"
            comment = (f"Reader View appears to significantly improve reading speed for individuals with dyslexia. "
                       f"Estimated multiplicative change in wpm = {pct_dys:.1f}% "
                       f"(95% CI {pct_dys_ci[0]:.1f}% to {pct_dys_ci[1]:.1f}%), p = {p_dys:.3g}.")
        elif (eff_dys > 0) and (p_dys >= 0.05):
            final_answer = "no_evidence"
            comment = (f"Reader View shows a positive (but not statistically significant) effect for dyslexic readers: "
                       f"estimated change = {pct_dys:.1f}% (95% CI {pct_dys_ci[0]:.1f}% to {pct_dys_ci[1]:.1f}%), "
                       f"p = {p_dys:.3g}.")
        elif (eff_dys <= 0) and (p_dys < 0.05):
            final_answer = "no_decrease"
            comment = (f"Reader View is associated with a statistically significant decrease (or no improvement) in "
                       f"reading speed for dyslexic readers: estimated change = {pct_dys:.1f}% "
                       f"(95% CI {pct_dys_ci[0]:.1f}% to {pct_dys_ci[1]:.1f}%), p = {p_dys:.3g}.")
        else:
            final_answer = "no"
            comment = (f"Reader View does not significantly improve reading speed for dyslexic readers: "
                       f"estimated change = {pct_dys:.1f}% (95% CI {pct_dys_ci[0]:.1f}% to {pct_dys_ci[1]:.1f}%), "
                       f"p = {p_dys:.3g}.")

    final_conclusion["answer"] = final_answer
    final_conclusion["comment"] = comment
    final_conclusion["decision_rule"] = ("We consider the effect 'improvement' if estimated log-wpm increase > 0 "
                                         "and the two-sided p-value for that marginal effect < 0.05.")

    description = (
        "Extracted statistics for the effect of ReaderView on log(wpm) and the ReaderView*Dyslexia interaction. "
        "See 'object' for coefficients, standard errors, p-values, 95% CIs, and percent multiplicative changes in wpm. "
        "Final conclusion (yes/no/inconclusive) on whether Reader View improves reading speed for individuals with "
        "dyslexia is provided below based on the dyslexic marginal effect."
    )

    return {
        "object": {
            "results": result_obj,
            "final_conclusion": final_conclusion
        },
        "description": description + " " + comment
    }