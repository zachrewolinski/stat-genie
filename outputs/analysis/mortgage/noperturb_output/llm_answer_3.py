def extract_final_answer(model_output):
    """
    Extracts the effect of the 'Female' indicator from a fitted statsmodels binary (Logit) results object.
    Returns a dict with numeric values under "object" and a short interpreted string under "description".
    """
    import numpy as np

    res = model_output

    # Validate expected attributes
    for attr in ('params', 'pvalues', 'bse', 'conf_int'):
        if not hasattr(res, attr):
            raise ValueError(f"Provided model output is missing required attribute: {attr}")

    params = res.params
    pvalues = res.pvalues
    bse = res.bse
    conf = res.conf_int()  # DataFrame/ndarray of log-odds CI

    if 'Female' not in params.index:
        raise KeyError("The model does not contain a parameter named 'Female'")

    # Extract statistics for Female
    coef = float(params['Female'])               # log-odds coefficient
    se = float(bse['Female'])
    pval = float(pvalues['Female'])

    # Odds ratio and 95% CI (exponentiate log-odds and CI)
    odds_ratio = float(np.exp(coef))

    # conf may be DataFrame (with .loc) or numpy array; handle both
    try:
        ci_low_log, ci_up_log = conf.loc['Female'][0], conf.loc['Female'][1]
    except Exception:
        # fallback for ndarray-like conf where rows follow params.index order
        idx = list(params.index).index('Female')
        ci_low_log, ci_up_log = conf[idx, 0], conf[idx, 1]

    ci_lower = float(np.exp(ci_low_log))
    ci_upper = float(np.exp(ci_up_log))

    # Percent change in odds and significance flag
    percent_change = (odds_ratio - 1.0) * 100.0
    significant_at_0_05 = bool(pval < 0.05)

    # Assemble returned object and human-readable description
    result_object = {
        'coefficient_log_odds': coef,
        'std_error': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'odds_ratio_95ci_lower': ci_lower,
        'odds_ratio_95ci_upper': ci_upper,
        'percent_change_in_odds': percent_change,
        'significant_at_0.05': significant_at_0_05
    }

    interpretation = (
        f"'Female' coefficient = {coef:.4f} (log-odds). Odds ratio = {odds_ratio:.3f} "
        f"(95% CI: {ci_lower:.3f} to {ci_upper:.3f}). p-value = {pval:.3g}. "
        f"Interpretation: holding the listed controls constant, being female is associated with a "
        f"{'higher' if odds_ratio>1 else 'lower' if odds_ratio<1 else 'no change in'} odds of mortgage acceptance "
        f"(change in odds = {percent_change:.1f}%). Statistically significant at α=0.05: "
        f"{'yes' if significant_at_0_05 else 'no'}."
    )

    return {"object": result_object, "description": interpretation}