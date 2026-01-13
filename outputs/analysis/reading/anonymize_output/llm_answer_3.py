def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on LogReadingSpeed specifically for
    readers with dyslexia (i.e., the sum of the ReaderView main effect and the
    ReaderView:Dyslexia interaction), along with standard error, t-stat, p-value,
    and 95% confidence interval. Also returns the ReaderView effect for non-dyslexic
    readers (the main effect alone) for comparison.

    Returns a dictionary with:
      - "object": dict of numeric results
      - "description": human-readable interpretation of the results and what they mean
                       for whether Reader View improves reading speed for dyslexic readers.
    """
    import numpy as np
    from scipy import stats

    results = model_output  # statsmodels RegressionResultsWrapper

    # Ensure we have params and covariance
    params = results.params
    cov = results.cov_params()  # uses the covariance used for inference (HC3 here)

    # Helper to find the interaction term name that contains both 'ReaderView' and 'Dyslexia'
    def find_term(containing):
        for name in params.index:
            if all(token in name for token in containing):
                return name
        return None

    # Locate expected terms
    term_reader = find_term(['ReaderView'])  # should match 'ReaderView'
    term_inter = find_term(['ReaderView', 'Dyslexia'])  # e.g., 'ReaderView:Dyslexia'
    term_dys = find_term(['Dyslexia'])  # main effect for Dyslexia (not strictly required)

    if term_reader is None:
        raise KeyError("Could not find a parameter name containing 'ReaderView' in model params.")
    if term_inter is None:
        # If there is no interaction term, we can still report the main effect only.
        interaction_exists = False
    else:
        interaction_exists = True

    # Coefficient for ReaderView when Dyslexia = 0 (non-dyslexic readers)
    beta_reader = params[term_reader]
    var_reader = cov.loc[term_reader, term_reader]

    # If interaction exists, compute combined effect for Dyslexia = 1
    if interaction_exists:
        beta_inter = params[term_inter]
        var_inter = cov.loc[term_inter, term_inter]
        cov_ri = cov.loc[term_reader, term_inter]
        # Combined coefficient = beta_reader + beta_inter
        coef_dys = beta_reader + beta_inter
        var_coef_dys = var_reader + var_inter + 2.0 * cov_ri
    else:
        # No interaction: effect is same for dyslexic and non-dyslexic readers
        coef_dys = beta_reader
        var_coef_dys = var_reader

    se_dys = np.sqrt(var_coef_dys)
    # t-stat and p-value using t-distribution with df_resid
    df_resid = results.df_resid if hasattr(results, "df_resid") else np.inf
    t_stat = coef_dys / se_dys if se_dys != 0 else np.nan
    # two-sided p-value
    if np.isfinite(df_resid):
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df_resid)
        t_crit = stats.t.ppf(1.0 - 0.025, df_resid)
    else:
        # fallback to normal
        p_value = 2.0 * stats.norm.sf(np.abs(t_stat))
        t_crit = stats.norm.ppf(1.0 - 0.025)

    ci_lower = coef_dys - t_crit * se_dys
    ci_upper = coef_dys + t_crit * se_dys

    # Also provide the non-dyslexic ReaderView effect for comparison
    se_reader = np.sqrt(var_reader)
    t_reader = beta_reader / se_reader if se_reader != 0 else np.nan
    if np.isfinite(df_resid):
        p_reader = 2.0 * stats.t.sf(np.abs(t_reader), df_resid)
    else:
        p_reader = 2.0 * stats.norm.sf(np.abs(t_reader))

    # For interpretability: approximate percent change on original (1 + WPM) scale:
    # Because DV = log1p(WPM) = log(1 + WPM), a change of delta in DV corresponds to
    # multiplicative change of exp(delta) on (1 + WPM). Approx percent change ~ (exp(delta)-1)*100.
    pct_change_dys = (np.exp(coef_dys) - 1.0) * 100.0
    pct_change_reader = (np.exp(beta_reader) - 1.0) * 100.0

    # Build the object to return
    object_out = {
        "term_reader_name": term_reader,
        "term_interaction_name": term_inter if interaction_exists else None,
        "dyslexic_readerview_effect": float(coef_dys),
        "dyslexic_readerview_se": float(se_dys),
        "dyslexic_readerview_t": float(t_stat),
        "dyslexic_readerview_p": float(p_value),
        "dyslexic_readerview_ci_95": (float(ci_lower), float(ci_upper)),
        "dyslexic_readerview_pct_change_approx": float(pct_change_dys),
        "non_dyslexic_readerview_effect": float(beta_reader),
        "non_dyslexic_readerview_se": float(se_reader),
        "non_dyslexic_readerview_t": float(t_reader),
        "non_dyslexic_readerview_p": float(p_reader),
        "non_dyslexic_readerview_pct_change_approx": float(pct_change_reader),
        "df_resid": float(df_resid)
    }

    # Short interpretation: decide whether there's evidence that Reader View improves reading speed
    alpha = 0.05
    if np.isnan(p_value):
        conclusion = "Could not compute p-value for the dyslexic subgroup effect."
    else:
        if (coef_dys > 0) and (p_value < alpha):
            conclusion = (
                "Statistically significant evidence (p = {:.3g}) that Reader View increases "
                "log-reading-speed for readers with dyslexia. Estimated effect = {:+.4f} "
                "(approx. {:+.2f}% change on (1+WPM) scale). 95% CI = [{:+.4f}, {:+.4f}]."
            ).format(p_value, coef_dys, pct_change_dys, ci_lower, ci_upper)
        elif (coef_dys < 0) and (p_value < alpha):
            conclusion = (
                "Statistically significant evidence (p = {:.3g}) that Reader View decreases "
                "log-reading-speed for readers with dyslexia. Estimated effect = {:+.4f} "
                "(approx. {:+.2f}% change on (1+WPM) scale). 95% CI = [{:+.4f}, {:+.4f}]."
            ).format(p_value, coef_dys, pct_change_dys, ci_lower, ci_upper)
        else:
            conclusion = (
                "No statistically significant effect of Reader View on log-reading-speed for readers "
                "with dyslexia (p = {:.3g}). Estimated effect = {:+.4f} "
                "(approx. {:+.2f}% change on (1+WPM) scale). 95% CI = [{:+.4f}, {:+.4f}]."
            ).format(p_value, coef_dys, pct_change_dys, ci_lower, ci_upper)

    description = (
        "Extracted estimate of the effect of Reader View for dyslexic readers (i.e., "
        "ReaderView + ReaderView:Dyslexia interaction). 'object' contains numeric estimates: "
        "coefficient on the log(1+WPM) scale, robust SE, t-stat, two-sided p-value, and 95% CI, "
        "plus an approximate percent change on the (1+WPM) scale. Also included is the "
        "ReaderView effect for non-dyslexic readers for comparison. Interpretation: "
        + conclusion
    )

    return {"object": object_out, "description": description}