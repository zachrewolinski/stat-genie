def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, 95% CI, and interprets the effect
    of STR_log (student-teacher ratio, log-transformed) on AvgScore from a
    statsmodels RegressionResultsWrapper object.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results and boolean flags
      - "description": a human-readable interpretation of the results

    Example returned numeric fields:
      coef, std_err, t_value, p_value, conf_int_95 (list), significant_at_0.05 (bool),
      direction ("negative"/"positive"/"zero"), lower_STR_associated_with_higher_performance (bool),
      effect_1pct_decrease (float, change in AvgScore for 1% decrease in STR),
      effect_10pct_decrease (float, change in AvgScore for 10% decrease in STR),
      r_squared (if available, else None)
    """
    import numpy as np

    result = model_output  # statsmodels RegressionResultsWrapper

    varname = 'STR_log'
    # Check that the variable is present
    params = getattr(result, 'params', None)
    if params is None or varname not in params.index:
        return {
            "object": None,
            "description": f"Variable '{varname}' not found in model output. Available params: {list(params.index) if params is not None else 'None'}"
        }

    # Extract basic statistics
    coef = float(params[varname])
    # standard error, t-value, p-value
    std_err = float(result.bse[varname]) if hasattr(result, 'bse') else None
    t_value = float(result.tvalues[varname]) if hasattr(result, 'tvalues') else None
    p_value = float(result.pvalues[varname]) if hasattr(result, 'pvalues') else None

    # Confidence interval (try robust extraction)
    try:
        ci = result.conf_int()  # might be DataFrame or ndarray
        if hasattr(ci, 'loc'):
            ci_low, ci_high = map(float, ci.loc[varname])
        else:
            # assume ndarray; find index of varname in params
            idx = list(result.params.index).index(varname)
            ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        ci_low, ci_high = None, None

    # R-squared if available
    r_squared = float(result.rsquared) if hasattr(result, 'rsquared') else None

    # Interpretation helpers
    significant = (p_value is not None) and (p_value < 0.05)
    if abs(coef) < 1e-12:
        direction = "zero"
    elif coef > 0:
        direction = "positive"
    else:
        direction = "negative"

    # Lower STR (fewer students per teacher) corresponds to a decrease in STR.
    # If coef < 0 then increasing STR reduces AvgScore, so lowering STR increases AvgScore.
    lower_STR_associated_with_higher_performance = (coef < 0)

    # Compute effect sizes for a 1% and 10% decrease in STR.
    # A p% decrease corresponds to multiplicative factor (1 - p), delta_log = ln(1 - p).
    # For a 1% decrease:
    delta_log_1pct = np.log(0.99)   # ~ -0.01005
    effect_1pct_decrease = coef * delta_log_1pct
    # For a 10% decrease:
    delta_log_10pct = np.log(0.90)  # ~ -0.10536
    effect_10pct_decrease = coef * delta_log_10pct

    # Build human-readable description
    desc_parts = []
    desc_parts.append(f"Estimated coefficient on {varname}: {coef:.4f}")
    if std_err is not None:
        desc_parts.append(f"(SE = {std_err:.4f})")
    if t_value is not None:
        desc_parts.append(f"(t = {t_value:.3f})")
    if p_value is not None:
        desc_parts.append(f"p = {p_value:.4f}")
    if ci_low is not None and ci_high is not None:
        desc_parts.append(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

    interpretation = " ".join(desc_parts) + ". "

    # Interpret direction and significance
    if direction == "negative":
        interpretation += ("The negative coefficient indicates that higher student-teacher ratios "
                           "(more students per teacher) are associated with lower district AvgScore; "
                           "conversely, lower student-teacher ratios are associated with higher AvgScore. ")
    elif direction == "positive":
        interpretation += ("The positive coefficient indicates that higher student-teacher ratios "
                           "(more students per teacher) are associated with higher district AvgScore; "
                           "conversely, lower student-teacher ratios are associated with lower AvgScore. ")
    else:
        interpretation += "The estimated effect is essentially zero. "

    if significant:
        interpretation += "This effect is statistically significant at the 0.05 level. "
    else:
        interpretation += "This effect is not statistically significant at the 0.05 level. "

    interpretation += (f"Numerically, a 1% decrease in STR is associated with a change in AvgScore of "
                       f"{effect_1pct_decrease:.4f} points, and a 10% decrease in STR with a change of "
                       f"{effect_10pct_decrease:.4f} points (both computed using exact ln(1 - p) scaling). ")

    if r_squared is not None:
        interpretation += f"Model R-squared = {r_squared:.3f}. "

    # Prepare the object to return
    out_obj = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "conf_int_95": [ci_low, ci_high],
        "significant_at_0.05": significant,
        "direction": direction,
        "lower_STR_associated_with_higher_performance": lower_STR_associated_with_higher_performance,
        "effect_1pct_decrease": float(effect_1pct_decrease),
        "effect_10pct_decrease": float(effect_10pct_decrease),
        "r_squared": r_squared
    }

    return {
        "object": out_obj,
        "description": interpretation
    }