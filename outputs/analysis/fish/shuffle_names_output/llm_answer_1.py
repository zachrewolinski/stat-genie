def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels GLMResultsWrapper fitted with a log link
    and an offset (hours). Assumes model predictors include a constant named 'const'.
    
    Returns a dictionary with:
      - "object": nested dict with coefficients, SEs, p-values, 95% CIs (on coef scale),
                  IRRs (exp(coef)), 95% CIs for IRRs, baseline rate per hour (exp(const))
      - "description": human-readable summary interpreting the baseline fish/hour and
                       the multiplicative effects (IRRs) of predictors.
    """
    import numpy as np

    res = model_output

    # Basic parameter estimates
    params = res.params  # pandas Series
    bse = getattr(res, "bse", None)
    pvalues = getattr(res, "pvalues", None)
    conf = res.conf_int()  # DataFrame with two columns [lower, upper]
    # Exponentiated coefficients (Incidence Rate Ratios) and their CIs
    irr = np.exp(params)
    irr_conf = np.exp(conf)

    # Prepare structured output
    terms = {}
    for name in params.index:
        terms[name] = {
            "coef": float(params[name]),
            "se": float(bse[name]) if bse is not None else None,
            "p_value": float(pvalues[name]) if pvalues is not None else None,
            "95ci_coef": [float(conf.loc[name, 0]), float(conf.loc[name, 1])],
            "irr": float(irr[name]),
            "95ci_irr": [float(irr_conf.loc[name, 0]), float(irr_conf.loc[name, 1])],
        }

    # Baseline rate per hour interpretation:
    # With offset = log_hours and log link: log(E[y]) = log(hours) + X*beta
    # => E[y]/hours = exp(X*beta). When predictors=0 (reference), baseline rate/hour = exp(const).
    baseline_rate = None
    baseline_ci = None
    if "const" in params.index:
        baseline_rate = float(np.exp(params["const"]))
        baseline_ci = [float(np.exp(conf.loc["const", 0])), float(np.exp(conf.loc["const", 1]))]

    # Optional model fallback warning (if Poisson was used as fallback, the original code attaches a warning)
    fallback_warning = getattr(res, "model_fallback_warning", None)

    # Build description text
    desc_lines = []
    if baseline_rate is not None:
        desc_lines.append(
            f"Estimated baseline fish caught per hour (reference group: livebait=0, child=0, "
            f"mean-centered group size and campers = 0): {baseline_rate:.3f} fish/hour "
            f"(95% CI: {baseline_ci[0]:.3f} to {baseline_ci[1]:.3f})."
        )
    else:
        desc_lines.append("No constant term named 'const' found; cannot report baseline rate per hour.")

    desc_lines.append("For each predictor, reported IRR = exp(coef). IRR > 1 means higher catch rate per hour; IRR < 1 means lower rate.")
    for term in terms:
        if term == "const":
            continue
        t = terms[term]
        desc_lines.append(
            f"{term}: coef={t['coef']:.4f}, IRR={t['irr']:.3f} "
            f"(95% CI: {t['95ci_irr'][0]:.3f} - {t['95ci_irr'][1]:.3f}), p={t['p_value']:.3g}"
        )

    if fallback_warning:
        desc_lines.append(f"Note: {fallback_warning}")

    description = "\n".join(desc_lines)

    result = {
        "object": {
            "terms": terms,
            "baseline_rate_per_hour": baseline_rate,
            "baseline_rate_95ci": baseline_ci,
            "model_aic": float(res.aic) if hasattr(res, "aic") else None,
            "model_deviance": float(res.deviance) if hasattr(res, "deviance") else None,
            "fallback_warning": fallback_warning,
        },
        "description": description,
    }

    return result