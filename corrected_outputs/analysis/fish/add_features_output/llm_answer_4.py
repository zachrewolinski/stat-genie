def extract_final_answer(model_output):
    """
    Extract interpretable statistics (coefficients, IRRs, p-values, 95% CIs)
    from the fitted model stored in model_output.

    Returns:
      {
        "object": {
          "model_used": "negative_binomial" or "poisson",
          "overdispersion": float,
          "baseline_rate_per_hour": float,   # exp(intercept)
          "effects": {
            varname: {
              "coef": float,        # log rate ratio
              "irr": float,         # exp(coef) = multiplicative change in fish/hour
              "pvalue": float,
              "ci_lower": float,    # 95% CI for IRR (lower)
              "ci_upper": float     # 95% CI for IRR (upper)
            }, ...
          }
        },
        "description": "short explanation of what the numbers mean"
      }
    """
    import numpy as np

    # Choose the preferred model: negative_binomial if present, else poisson
    nb = model_output.get('negative_binomial', None)
    poisson = model_output.get('poisson', None)
    overdispersion = model_output.get('overdispersion', None)

    if nb is not None:
        res = nb
        model_used = 'negative_binomial'
    elif poisson is not None:
        res = poisson
        model_used = 'poisson'
    else:
        return {
            "object": None,
            "description": "No model results found in model_output."
        }

    # Extract parameters and statistics
    try:
        params = res.params.copy()       # pandas Series
        pvalues = res.pvalues.copy()
        ci = res.conf_int()              # DataFrame with two columns [lower, upper] in parameter scale (log)
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract stats from model object: {e}"
        }

    # Determine intercept name
    intercept_name = None
    for n in params.index:
        if n.lower() in ('intercept', 'const'):
            intercept_name = n
            break
    if intercept_name is None:
        # fallback to first parameter
        intercept_name = params.index[0]

    # Build output for a set of primary predictors if present; include all if primary missing
    primary_vars = ['livebait', 'camper', 'total_persons', 'child', 'age']
    effects = {}

    # Decide which variables to report: intersection of primary_vars that exist, else all params except county dummies if many
    available_primary = [v for v in primary_vars if v in params.index]
    if available_primary:
        report_vars = available_primary
    else:
        # if none of the primary vars present (unlikely), report all non-county params (exclude terms starting with 'C(county)')
        report_vars = [v for v in params.index if not (v.startswith('C(county)') or 'county' in v)]
        # ensure intercept included
        if intercept_name not in report_vars:
            report_vars = [intercept_name] + report_vars

    # Ensure intercept is included in reported effects
    if intercept_name not in report_vars:
        report_vars = [intercept_name] + report_vars

    for v in report_vars:
        coef = float(params.loc[v])
        p = float(pvalues.loc[v]) if v in pvalues.index else None
        ci_row = ci.loc[v] if v in ci.index else None
        if ci_row is not None:
            # conf_int is on coefficient (log) scale; exponentiate to get IRR CI
            ci_lower = float(np.exp(ci_row.iloc[0]))
            ci_upper = float(np.exp(ci_row.iloc[1]))
        else:
            ci_lower = ci_upper = None
        irr = float(np.exp(coef))
        effects[v] = {
            "coef": coef,
            "irr": irr,
            "pvalue": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    # Baseline rate (fish per hour) implied by intercept when predictors = 0:
    intercept_coef = float(params.loc[intercept_name])
    baseline_rate = float(np.exp(intercept_coef))

    result_object = {
        "model_used": model_used,
        "overdispersion": float(overdispersion) if overdispersion is not None else None,
        "baseline_rate_per_hour": baseline_rate,
        "effects": effects
    }

    description_lines = [
        f"Model used: {model_used}. Overdispersion (from Poisson): {overdispersion:.3f}" if overdispersion is not None else f"Model used: {model_used}.",
        "Reported coefficients are log rate ratios (change in log fish-per-hour).",
        "IRR = exp(coef) is the multiplicative effect on fish-per-hour for a one-unit increase in the predictor (or presence vs absence for binaries).",
        "95% CI shown for IRR is the exponentiated CI of the coefficient.",
        "Baseline rate_per_hour is exp(intercept) and represents the predicted fish-per-hour when all predictors are zero (interpret with care: zero values may be out-of-sample for some covariates).",
        "Predictors shown are the primary variables of interest (livebait, camper, total_persons, child, age).",
        "Effects with IRR > 1 increase the expected fish/hour; IRR < 1 decrease it. Use p-values to assess statistical significance (conventional threshold p < 0.05)."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}