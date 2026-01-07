def extract_final_answer(model_output):
    """
    Extract statistics for the 'is_female' coefficient from a fitted logistic model output.
    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, p_value, odds_ratio, ci_lower, ci_upper, ci_finite, statistically_significant)
      - "description": human-readable interpretation of the effect of gender on approval
    """
    import numpy as np
    import pandas as pd

    # Validate input structure
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict with keys like 'model_result' or 'odds_ratios'.")

    # Try to get the statsmodels result if present
    result = model_output.get('model_result', None)

    # If statsmodels result available, extract from it; otherwise try to use the odds_ratios table
    coef = se = pval = None
    ci_lower = ci_upper = None
    odds_ratio = None

    if result is not None:
        # Expect a statsmodels BinaryResultsWrapper or similar
        try:
            params = result.params
            bse = result.bse
            pvalues = result.pvalues
            conf = result.conf_int()  # DataFrame with two columns
        except Exception as e:
            raise ValueError(f"Unable to extract parameters from model_result: {e}")

        if 'is_female' not in params.index:
            raise KeyError("Model result does not contain 'is_female' coefficient.")

        coef = float(params.loc['is_female'])
        se = float(bse.loc['is_female']) if 'is_female' in bse.index else None
        pval = float(pvalues.loc['is_female']) if 'is_female' in pvalues.index else None

        # confidence interval on log-odds scale
        if 'is_female' in conf.index:
            ci_lower_log = conf.loc['is_female'].iloc[0]
            ci_upper_log = conf.loc['is_female'].iloc[1]
            # convert to odds-ratio scale if finite
            try:
                ci_lower = np.exp(ci_lower_log) if np.isfinite(ci_lower_log) else np.nan
                ci_upper = np.exp(ci_upper_log) if np.isfinite(ci_upper_log) else np.nan
            except Exception:
                ci_lower = np.nan
                ci_upper = np.nan
        else:
            ci_lower = ci_upper = np.nan

        # odds ratio
        try:
            odds_ratio = float(np.exp(coef))
        except Exception:
            odds_ratio = None

    else:
        # fallback: try using odds_ratios DataFrame if provided
        odds_df = model_output.get('odds_ratios', None)
        if odds_df is None:
            raise KeyError("model_output must contain either 'model_result' or 'odds_ratios'.")
        if 'is_female' not in odds_df.index:
            raise KeyError("odds_ratios table does not contain 'is_female' row.")
        odds_ratio = float(odds_df.loc['is_female', 'odds_ratio'])
        ci_lower = float(odds_df.loc['is_female', 'ci_lower']) if pd.notna(odds_df.loc['is_female', 'ci_lower']) else np.nan
        ci_upper = float(odds_df.loc['is_female', 'ci_upper']) if pd.notna(odds_df.loc['is_female', 'ci_upper']) else np.nan
        # coefficient and p-value not available in this fallback
        coef = se = pval = None

    # Determine whether CI bounds are finite and informative
    ci_finite = np.isfinite(ci_lower) and np.isfinite(ci_upper) and (ci_lower > 0)

    # Determine statistical significance if p-value available and finite
    statistically_significant = None
    if pval is not None and np.isfinite(pval):
        statistically_significant = bool(pval < 0.05)

    # Direction of effect (based on odds_ratio if available)
    direction = None
    if odds_ratio is not None and np.isfinite(odds_ratio):
        if odds_ratio > 1:
            direction = "female applicants have higher odds of approval (OR > 1)"
        elif odds_ratio < 1:
            direction = "female applicants have lower odds of approval (OR < 1)"
        else:
            direction = "no difference in odds (OR = 1)"

    # Build object to return
    obj = {
        'coef_log_odds': coef,                   # log-odds coefficient (None if unavailable)
        'std_error': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'ci_lower_odds_ratio': ci_lower,
        'ci_upper_odds_ratio': ci_upper,
        'ci_finite': bool(ci_finite),
        'statistically_significant_at_0.05': statistically_significant,
        'direction': direction
    }

    # Build a concise description interpreting the result
    if odds_ratio is None:
        desc = "Could not compute odds ratio for 'is_female'."
    else:
        desc_parts = []
        desc_parts.append(f"The estimated odds ratio for is_female is {odds_ratio:.3g} "
                          f"({'>1' if odds_ratio>1 else '<1' if odds_ratio<1 else '=1'}).")
        if direction is not None:
            desc_parts.append(direction + ".")
        if statistically_significant is True:
            desc_parts.append(f"The coefficient is statistically significant (p = {pval:.3g}).")
        elif statistically_significant is False:
            desc_parts.append(f"The coefficient is not statistically significant (p = {pval:.3g}).")
        else:
            # p-value unavailable
            if pval is None:
                desc_parts.append("No p-value available to assess statistical significance.")
            else:
                desc_parts.append(f"p-value = {pval:.3g}.")

        if ci_finite:
            desc_parts.append(f"The 95% CI for the odds ratio is [{ci_lower:.3g}, {ci_upper:.3g}].")
        else:
            desc_parts.append("The 95% CI for the odds ratio is not finite or not informative (e.g., [0, ∞]). "
                              "This suggests potential separation/convergence issues or extremely wide uncertainty; "
                              "interpret effect size with caution.")

        desc = " ".join(desc_parts)

    return {"object": obj, "description": desc}