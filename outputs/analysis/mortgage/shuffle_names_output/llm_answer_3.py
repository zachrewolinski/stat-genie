def extract_final_answer(model_output):
    """
    Extract statistics for the 'is_female' coefficient from a fitted statsmodels
    binary outcome model (Logit or GLM results wrapper).

    Returns a dict with:
      - "object": dict with numeric results: coef, se, pvalue, CI (log-odds),
                  odds_ratio, odds_ratio_CI, and boolean 'significant' at alpha=0.05
      - "description": brief plain-language interpretation in the context of
                       whether being female affects mortgage denial probability.
    """
    import numpy as np

    res = model_output

    # Helper to raise a clear error if the coefficient isn't present
    if 'is_female' not in getattr(res, 'params').index:
        raise KeyError("Model output does not contain a parameter named 'is_female'.")

    # Extract point estimate, SE, p-value
    coef = float(res.params['is_female'])
    se = float(res.bse['is_female'])
    pvalue = float(res.pvalues['is_female'])

    # Extract confidence interval for the coefficient (log-odds scale)
    try:
        ci = res.conf_int()  # DataFrame or ndarray-like
        # If it's a DataFrame with index, use .loc
        if hasattr(ci, 'loc'):
            ci_low, ci_high = float(ci.loc['is_female', 0]), float(ci.loc['is_female', 1])
        else:
            # ndarray: find the row index of is_female in params.index
            row_idx = list(res.params.index).index('is_female')
            ci_low, ci_high = float(ci[row_idx, 0]), float(ci[row_idx, 1])
    except Exception as e:
        # As a fallback, approximate CI using coef +/- 1.96*SE
        ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Significance at conventional alpha
    alpha = 0.05
    significant = pvalue < alpha

    # Interpretation in context
    if coef > 0:
        direction = "higher"
    elif coef < 0:
        direction = "lower"
    else:
        direction = "no difference"

    description = (
        f"The model coefficient for 'is_female' is {coef:.4f} (SE={se:.4f}), p={pvalue:.4g}. "
        f"The {100*(1-alpha):.0f}% confidence interval on the log-odds scale is "
        f"[{ci_low:.4f}, {ci_high:.4f}].\n"
        f"On the odds-ratio scale, the estimated odds ratio is {odds_ratio:.3f} "
        f"(CI [{or_ci_low:.3f}, {or_ci_high:.3f}]).\n"
        f"This means females have {direction} odds of mortgage denial compared to males "
        f"(odds ratio {'>' if odds_ratio>1 else '<' if odds_ratio<1 else '='} 1). "
        f"The effect is {'statistically significant' if significant else 'not statistically significant'} "
        f"at alpha = {alpha}."
    )

    result_object = {
        "coef_log_odds": coef,
        "se": se,
        "p_value": pvalue,
        "ci_log_odds": [ci_low, ci_high],
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": [or_ci_low, or_ci_high],
        "significant_at_0.05": bool(significant),
        "alpha": alpha
    }

    return {"object": result_object, "description": description}