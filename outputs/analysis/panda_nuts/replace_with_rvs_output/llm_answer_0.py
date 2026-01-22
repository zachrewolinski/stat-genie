def extract_final_answer(model_output):
    """
    Extract coefficient, SE, p-value, and 95% CI for the predictors of interest
    (age, sex_M, help_yes) from a fitted statsmodels model object (MixedLMResultsWrapper,
    OLS result, or a clustered-robust wrapper).

    Returns:
      dict with keys:
        - "object": dict mapping each predictor to a dict of extracted stats (or None
                    if predictor not present)
        - "description": human-readable summary of the estimates and their interpretation
    """
    import math
    import numpy as np
    import pandas as pd

    # Try to get fixed-effect parameter estimates
    if hasattr(model_output, 'fe_params'):
        params = pd.Series(model_output.fe_params)
    elif hasattr(model_output, 'params'):
        params = pd.Series(model_output.params)
    else:
        raise ValueError("Model output does not expose params or fe_params.")

    # Try to get standard errors
    bse = None
    if hasattr(model_output, 'bse_fe'):
        bse = pd.Series(model_output.bse_fe)
    elif hasattr(model_output, 'bse'):
        try:
            bse = pd.Series(model_output.bse)
        except Exception:
            bse = None

    # If bse not available, try to compute from covariance matrix
    if bse is None:
        try:
            cov = model_output.cov_params()
            # cov_params() might return a DataFrame or ndarray
            cov_arr = np.asarray(cov)
            se_arr = np.sqrt(np.diag(cov_arr))
            # Try to align indices if cov was a DataFrame
            if hasattr(cov, 'index'):
                bse = pd.Series(se_arr, index=cov.index)
            else:
                bse = pd.Series(se_arr, index=params.index)
        except Exception:
            # give up and fill with NaNs
            bse = pd.Series(np.nan, index=params.index)

    # Try to get p-values; otherwise compute via normal approximation
    if hasattr(model_output, 'pvalues'):
        try:
            pvalues = pd.Series(model_output.pvalues)
        except Exception:
            pvalues = None
    else:
        pvalues = None

    if pvalues is None:
        # Compute z and two-sided p using standard normal (erfc) without requiring scipy
        # two-sided p = erfc(|z|/sqrt(2))
        z = params / bse
        pvals_dict = {}
        for name, zval in z.items():
            try:
                pvals_dict[name] = math.erfc(abs(float(zval)) / math.sqrt(2.0))
            except Exception:
                pvals_dict[name] = float('nan')
        pvalues = pd.Series(pvals_dict)

    # 95% CI using normal critical value ~1.96
    ci_low = params - 1.96 * bse
    ci_high = params + 1.96 * bse

    # Variables of interest
    predictors = ['age', 'sex_M', 'help_yes']

    results = {}
    summary_lines = []
    for var in predictors:
        if var in params.index:
            coef = float(params.loc[var]) if not pd.isna(params.loc[var]) else float('nan')
            se = float(bse.loc[var]) if (var in bse.index and not pd.isna(bse.loc[var])) else float('nan')
            p = float(pvalues.loc[var]) if (var in pvalues.index and not pd.isna(pvalues.loc[var])) else float('nan')
            ci = (float(ci_low.loc[var]) if (var in ci_low.index and not pd.isna(ci_low.loc[var])) else float('nan'),
                  float(ci_high.loc[var]) if (var in ci_high.index and not pd.isna(ci_high.loc[var])) else float('nan'))
            direction = 'positive' if coef > 0 else ('negative' if coef < 0 else 'none')
            significance = ('p < 0.05' if (not math.isnan(p) and p < 0.05) else 'p >= 0.05 or unknown')
            results[var] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                'ci_95': ci,
                'direction': direction,
                'significance': significance
            }
            summary_lines.append(
                f"{var}: coef={coef:.4g}, se={se:.4g}, p={p:.3g}, 95%CI=({ci[0]:.4g}, {ci[1]:.4g}) -> {direction}, {significance}."
            )
        else:
            results[var] = None
            summary_lines.append(f"{var}: NOT in model (no estimate).")

    # Short interpretation in the context of nut-cracking efficiency
    interpretation = (
        "Fixed-effect estimates for predictors of nut-cracking efficiency (nuts_per_sec). "
        "Each line reports coefficient (change in nuts/sec per one-unit change in predictor), "
        "standard error, two-sided p-value (normal approximation if not provided by model), "
        "95% confidence interval, direction of effect, and whether p < 0.05.\n\n"
        + "\n".join(summary_lines)
        + "\n\nNotes: Positive coefficient means higher nut-cracking efficiency associated with higher predictor value "
        "(older age, being male if sex_M == 1, or having received help if help_yes == 1). "
        "Random-intercept variance (between-chimpanzee) is not repeated here but remains in the model."
    )

    return {"object": results, "description": interpretation}