def extract_final_answer(model_output):
    """
    Extract coefficient, standard error, t-value/z-value, p-value, 95% CI, and
    multiplicative effect (exp(coef)-1) for the predictors of interest:
      - age_years
      - sex_male
      - help_received

    Returns:
      {
        "object": {
          "age_years": { "coef": ..., "se": ..., "t_or_z": ..., "p_value": ..., "ci_lower": ..., "ci_upper": ..., "multiplicative_change": ... },
          "sex_male": { ... },
          "help_received": { ... }
        },
        "description": "Concise interpretation string ..."
      }
    """
    import numpy as np

    res = model_output

    # Basic validation
    if not hasattr(res, "params"):
        raise ValueError("Provided model_output does not look like a fitted statsmodels results object (missing .params).")

    predictors = ['age_years', 'sex_male', 'help_received']

    params = res.params
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    tvalues = getattr(res, 'tvalues', None)

    # Attempt to get confidence intervals
    try:
        ci = res.conf_int()
    except Exception:
        ci = None

    results = {}
    summary_parts = []

    for var in predictors:
        if var not in params.index:
            results[var] = None
            summary_parts.append(f"{var}: not found in the model.")
            continue

        coef = float(params[var])
        se = float(bse[var]) if (bse is not None and var in bse.index) else None
        pval = float(pvalues[var]) if (pvalues is not None and var in pvalues.index) else None
        t_or_z = float(tvalues[var]) if (tvalues is not None and var in tvalues.index) else None

        # Confidence interval extraction robust to different types returned
        ci_lower = ci_upper = None
        if ci is not None:
            try:
                # If ci is a DataFrame with index matching params.index
                ci_lower = float(ci.loc[var, 0])
                ci_upper = float(ci.loc[var, 1])
            except Exception:
                try:
                    # If ci is a numpy array with same order as params.index
                    idx = list(params.index).index(var)
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
                except Exception:
                    ci_lower = ci_upper = None

        # Multiplicative change in original efficiency scale (approx percent change)
        multiplicative_change = float(np.exp(coef) - 1.0)

        results[var] = {
            "coef": coef,
            "se": se,
            "t_or_z": t_or_z,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "multiplicative_change": multiplicative_change  # e.g., 0.10 means ~+10% efficiency
        }

        # Short human-readable summary per predictor
        if pval is None:
            sig_text = "p-value unavailable"
        else:
            sig_text = "statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)"
        direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
        pct = multiplicative_change * 100.0
        summary_parts.append(
            f"{var}: coef={coef:.4f} ({direction}); {sig_text}; approx {pct:.1f}% change in raw efficiency per unit."
        )

    description = (
        "Extracted model statistics for predictors of nut-cracking log-efficiency. "
        "Coefficients are on the log(nuts_opened/session_seconds) scale; exp(coef)-1 gives the "
        "approximate proportional change in raw efficiency (e.g., 0.10 = +10%). "
        "Summary: " + " ".join(summary_parts)
    )

    return {"object": results, "description": description}