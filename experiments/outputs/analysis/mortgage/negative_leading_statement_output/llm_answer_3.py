def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator on mortgage acceptance from a fitted model_output.
    Returns a dictionary with:
      - "object": dict with numeric results (coefficient if available, odds ratio, 95% CI, p-value, percent change, significance)
      - "description": plain-language interpretation of the female effect in context (controls listed)
    The function accepts the model_output dict produced by the modeling function (with keys 'fitted_model' and/or 'odds_ratios').
    """
    import numpy as np
    import pandas as pd

    def _find_row(df, name):
        # Try to locate a row by exact label or case-insensitive match
        if name in df.index:
            return df.loc[name]
        lname = name.lower()
        for idx in df.index:
            try:
                if str(idx).lower() == lname:
                    return df.loc[idx]
            except Exception:
                continue
        return None

    # Initialize outputs
    coef = None
    or_val = None
    ci_low = None
    ci_up = None
    p = None

    # If odds_ratios table is available, prefer it for OR/CI/pvalue
    if isinstance(model_output, dict) and 'odds_ratios' in model_output:
        or_table = model_output['odds_ratios']
        row = _find_row(or_table, 'female')
        if row is not None:
            try:
                or_val = float(row['OR'])
                ci_low = float(row['OR_2.5%'])
                ci_up = float(row['OR_97.5%'])
                p = float(row['pvalue'])
            except Exception:
                # fallback to None on parse error
                or_val = ci_low = ci_up = p = None

    # If fitted_model is available, extract coefficient and (if needed) compute OR/CI/pvalue
    if isinstance(model_output, dict) and 'fitted_model' in model_output:
        res = model_output['fitted_model']
        # try to get coefficient and p-value from fitted_model
        try:
            coef = float(res.params.get('female', np.nan))
        except Exception:
            coef = None
        try:
            p_from_res = float(res.pvalues.get('female', np.nan))
        except Exception:
            p_from_res = None

        # compute OR and CI from fitted_model if not obtained from odds table
        if or_val is None or ci_low is None or ci_up is None or p is None:
            try:
                # conf_int returns DataFrame with two columns; find row for 'female'
                conf = res.conf_int()
                conf_row = _find_row(conf, 'female')
                if conf_row is not None:
                    ci_low = float(np.exp(conf_row.iloc[0]))
                    ci_up = float(np.exp(conf_row.iloc[1]))
                if coef is not None and not np.isnan(coef):
                    or_val = float(np.exp(coef))
                # prefer p from fitted model if we didn't get it earlier
                if p is None:
                    p = p_from_res
            except Exception:
                # leave values as-is if any step fails
                pass
        else:
            # if odds table provided p but coef is missing, keep p and coef
            if p is None:
                p = p_from_res

    # Final sanity checks: if still None, raise informative error
    if or_val is None or p is None:
        raise ValueError("Could not extract odds ratio and p-value for 'female' from model_output. "
                         "Ensure model_output contains 'odds_ratios' DataFrame or a 'fitted_model' with params/conf_int/pvalues.")

    # Compute percent change in odds and significance flag
    percent_change = (or_val - 1.0) * 100.0
    significant = bool(p < 0.05)

    # Build output object and description
    object_out = {
        'coefficient_log_odds': coef,           # may be None if not available
        'odds_ratio': or_val,
        'OR_95%_CI_lower': ci_low,
        'OR_95%_CI_upper': ci_up,
        'p_value': p,
        'percent_change_in_odds': percent_change,
        'significant_at_0.05': significant
    }

    direction = 'higher' if percent_change > 0 else ('lower' if percent_change < 0 else 'no difference')
    description = (
        f"Holding the listed controls constant (black, mortgage_credit_z, consumer_credit_z, PI_ratio_z, "
        f"loan_to_value_z, bad_history, housing_expense_ratio_z, self_employed, married, denied_PMI), "
        f"the estimated odds ratio for female (vs male) is {or_val:.3f} "
        f"(95% CI: {ci_low:.3f} to {ci_up:.3f}), p = {p:.3g}. "
        f"This implies female applicants have {abs(percent_change):.1f}% {direction} odds of mortgage approval compared to male applicants. "
        f"The effect is {'statistically significant (p < 0.05)' if significant else 'not statistically significant (p >= 0.05)'}."
    )

    return {"object": object_out, "description": description}