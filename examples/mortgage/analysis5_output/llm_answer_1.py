def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'female' from a fitted statsmodels Logit result
    (BinaryResultsWrapper).

    Returns a dictionary with keys:
      - "object": a dict containing numeric results for the 'female' coefficient
      - "description": a short human-readable interpretation of these numbers

    The "object" dict contains:
      - coef: estimated log-odds coefficient for 'female'
      - se: standard error of the coefficient
      - p_value: two-sided p-value for the coefficient
      - ci_lower, ci_upper: 95% confidence interval for the coefficient (log-odds)
      - odds_ratio: exp(coef) giving multiplicative change in odds for female vs male
      - odds_ratio_ci: 95% CI for odds ratio (exp of coefficient CI)
      - avg_probability_increase_if_female: average predicted probability difference
        (Pr(accept | female=1) - Pr(accept | female=0)) holding other covariates at their observed values
        (None if this cannot be computed)
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("Provided model_output does not look like a fitted statsmodels results object (missing .params).")

    params = res.params
    pvalues = res.pvalues
    bse = res.bse
    try:
        conf = res.conf_int()
    except Exception:
        # fallback: compute approximate conf interval from coef +/- 1.96*se
        conf = pd.DataFrame({
            0: params - 1.96 * bse,
            1: params + 1.96 * bse
        }, index=params.index)

    if 'female' not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'female'.")

    coef = float(params['female'])
    se = float(bse['female']) if 'female' in bse.index else None
    p_value = float(pvalues['female']) if 'female' in pvalues.index else None
    ci_lower = float(conf.loc['female'].iloc[0])
    ci_upper = float(conf.loc['female'].iloc[1])

    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = [float(np.exp(ci_lower)), float(np.exp(ci_upper))]

    # Attempt to compute average discrete change in predicted probability when female changes 0->1
    avg_prob_diff = None
    try:
        # Get original exog matrix and column names used to fit the model
        exog = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
        if 'female' in exog.columns:
            exog_f1 = exog.copy()
            exog_f0 = exog.copy()
            exog_f1['female'] = 1
            exog_f0['female'] = 0
            # Use model's predict method
            pred1 = res.predict(exog_f1)
            pred0 = res.predict(exog_f0)
            # Ensure numeric array (in case pandas Series)
            pred1 = np.asarray(pred1, dtype=float)
            pred0 = np.asarray(pred0, dtype=float)
            avg_prob_diff = float((pred1 - pred0).mean())
    except Exception:
        avg_prob_diff = None

    result_object = {
        'coef': coef,
        'se': se,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'odds_ratio': odds_ratio,
        'odds_ratio_ci': odds_ratio_ci,
        'avg_probability_increase_if_female': avg_prob_diff
    }

    # Create a concise interpretation
    desc_lines = []
    desc_lines.append(f"The logistic regression coefficient for 'female' is {coef:.4f} (SE={se:.4f}), "
                      f"p={p_value:.4g}).")
    desc_lines.append(f"This corresponds to an odds ratio of {odds_ratio:.3f} "
                      f"(95% CI [{odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f}]).")
    if avg_prob_diff is not None:
        desc_lines.append(f"On average, setting female=1 vs female=0 changes the predicted approval probability by "
                          f"{avg_prob_diff:.4f} (in probability points), holding other covariates at their observed values).")
    else:
        desc_lines.append("Average predicted probability change could not be computed from the model object.")
    desc_lines.append("Interpretation: the coefficient is on the log-odds scale. A positive coef (or odds ratio > 1) "
                      "means female applicants have higher odds of approval than male applicants, conditional on controls; "
                      "a negative coef (odds ratio < 1) means lower odds. Statistical significance is indicated by the p-value.")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}