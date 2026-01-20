def extract_final_answer(model_output):
    """
    Extracts the effect of the IsHuman indicator from a fitted statsmodels GLMResultsWrapper
    (binomial GLM). Returns a dictionary with a numeric summary under "object" and a short
    interpretive text under "description".

    The "object" value is itself a dict with:
      - coef_log_odds: coefficient for IsHuman (log-odds scale)
      - se: standard error of the coefficient
      - z_value: test statistic (coef / se)
      - p_value: two-sided p-value for the coefficient
      - ci_log_odds: 95% confidence interval for the coefficient (tuple: lower, upper)
      - odds_ratio: exp(coef) (multiplicative change in odds for IsHuman=1 vs 0)
      - odds_ratio_ci: 95% CI for the odds ratio (tuple: lower, upper)
      - humans_higher: boolean, True if coef>0 and p_value < 0.05 (evidence that humans have higher AMTL)
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Try to find the IsHuman parameter name in the results index
    params_index = list(res.params.index)
    matches = [name for name in params_index if name == 'IsHuman' or name.lower().endswith('ishuman')]
    if len(matches) == 0:
        # try substring match as a last resort
        matches = [name for name in params_index if 'ishuman' in name.lower() or 'is_human' in name.lower()]

    if len(matches) == 0:
        return {
            "object": None,
            "description": "The fitted model does not contain a parameter named 'IsHuman' (no matching column found)."
        }

    param_name = matches[0]

    coef = float(res.params[param_name])
    se = float(res.bse[param_name]) if hasattr(res, 'bse') else float(res.HC0_se[param_name])
    # compute z/t value robustly
    z_value = coef / se if se != 0 else np.nan
    p_value = float(res.pvalues[param_name]) if hasattr(res, 'pvalues') else float(res.pvalue[param_name])

    # 95% CI on log-odds
    try:
        ci_df = res.conf_int()
        ci_lower, ci_upper = float(ci_df.loc[param_name, 0]), float(ci_df.loc[param_name, 1])
    except Exception:
        # fallback: use coef +/- 1.96*se
        ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se

    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))

    # Decision rule: positive coef and statistically significant at alpha=0.05
    humans_higher = (coef > 0) and (p_value < 0.05)

    result_object = {
        "param_name": param_name,
        "coef_log_odds": coef,
        "se": se,
        "z_value": z_value,
        "p_value": p_value,
        "ci_log_odds": (ci_lower, ci_upper),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": odds_ratio_ci,
        "humans_higher": humans_higher
    }

    # Build concise interpretation
    if humans_higher:
        interpretation = (
            f"The IsHuman coefficient is positive (log-odds = {coef:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]), "
            f"odds ratio = {odds_ratio:.3f} (95% CI [{odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f}]), "
            f"and statistically significant (p = {p_value:.4g}). This indicates that, after controlling for age, sex, "
            "and tooth class, modern humans (Homo sapiens) have a higher frequency of antemortem tooth loss compared to the non-human genera in the dataset."
        )
    else:
        interpretation = (
            f"The IsHuman coefficient is {('positive' if coef>0 else 'negative')} (log-odds = {coef:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]), "
            f"odds ratio = {odds_ratio:.3f} (95% CI [{odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f}]), "
            f"with p = {p_value:.4g}. This does not provide statistically significant evidence (at alpha=0.05) that modern humans have higher AMTL after accounting "
            "for age, sex, and tooth class."
        )

    return {
        "object": result_object,
        "description": interpretation
    }