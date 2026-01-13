def extract_final_answer(model_output):
    """
    Extracts coefficient estimates, p-values, confidence intervals, and rate ratios
    from a fitted count model (Poisson or Negative Binomial) that used log(hours)
    as an offset.

    Returns:
      {
        "object": {
           "<var>": {
               "coef": float,               # log-rate coefficient (per hour)
               "se": float,                 # standard error
               "pvalue": float,
               "ci_lower": float,           # 95% CI on log scale
               "ci_upper": float,
               "rate_ratio": float,         # exp(coef): multiplicative effect on fish/hour
               "rr_ci_lower": float,        # exp(ci_lower)
               "rr_ci_upper": float         # exp(ci_upper)
           },
           ...
           "model_used": str,
           "dispersion": float or None
        },
        "description": str  # short plain-language interpretation
      }
    """
    import numpy as np

    # Choose the best available fitted model (prefer negative binomial if present)
    model = None
    if model_output is None:
        raise ValueError("model_output is None")
    # Prefer explicitly chosen model if indicated
    used = model_output.get('used_model', None)
    # Try negative_binomial first if present and not None
    if model_output.get('negative_binomial') is not None:
        model = model_output['negative_binomial']
    elif model_output.get('poisson') is not None:
        model = model_output['poisson']
    else:
        # Fallback: try any value that looks like a results object
        for k in ['negative_binomial', 'poisson']:
            if k in model_output and model_output[k] is not None:
                model = model_output[k]
                break
    if model is None:
        raise ValueError("No fitted model found in model_output")

    # Try to obtain parameter names
    try:
        params = model.params         # pandas Series usually
        pvalues = model.pvalues
        bse = model.bse
        conf = model.conf_int()       # DataFrame/ndarray with two columns
    except Exception as e:
        raise ValueError(f"Unable to extract parameters from model object: {e}")

    # Ensure we have a names list
    try:
        names = list(params.index)
    except Exception:
        # Try model.model.exog_names
        try:
            names = list(model.model.exog_names)
        except Exception:
            # Fallback to assumed names
            names = ['const', 'livebait', 'camper', 'total_people']

    # Build summary for each parameter
    summary = {}
    for i, name in enumerate(names):
        coef = float(params.loc[name]) if hasattr(params, "loc") else float(params[i])
        se = float(bse.loc[name]) if hasattr(bse, "loc") else float(bse[i])
        pval = float(pvalues.loc[name]) if hasattr(pvalues, "loc") else float(pvalues[i])
        # conf could be DataFrame or ndarray
        try:
            ci_row = conf.loc[name] if hasattr(conf, "loc") else conf[i]
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
        except Exception:
            # Fallback: use coef +/- 1.96*se
            ci_lower = float(coef - 1.96 * se)
            ci_upper = float(coef + 1.96 * se)

        rr = float(np.exp(coef))
        rr_ci_lower = float(np.exp(ci_lower))
        rr_ci_upper = float(np.exp(ci_upper))

        summary[name] = {
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "rate_ratio": rr,
            "rr_ci_lower": rr_ci_lower,
            "rr_ci_upper": rr_ci_upper
        }

    # Add model-level info
    summary["model_used"] = used if used is not None else (
        "negative_binomial" if model_output.get('negative_binomial') is not None else "poisson"
    )
    summary["dispersion"] = float(model_output.get('dispersion')) if model_output.get('dispersion') is not None else None

    # Compose a brief description interpreting the coefficients
    # Note: Because the model used offset = log_hours, coefficients are log(rate per hour).
    desc_lines = []
    desc_lines.append(f"Model used: {summary['model_used']}.")
    if summary["dispersion"] is not None:
        desc_lines.append(f"Dispersion (Pearson chi2/df): {summary['dispersion']:.3g}.")
    desc_lines.append("Coefficients are on the log-rate (per hour) scale because log(hours) was used as an offset.")
    desc_lines.append("Exponentiated coefficients (rate_ratio) are multiplicative effects on fish caught per hour, holding other variables constant.")
    # Interpret key variables if present
    for var in ['livebait', 'camper', 'total_people']:
        if var in summary:
            rr = summary[var]["rate_ratio"]
            rr_ci_lo = summary[var]["rr_ci_lower"]
            rr_ci_hi = summary[var]["rr_ci_upper"]
            pval = summary[var]["pvalue"]
            # Plain language
            if var == 'livebait':
                desc_lines.append(
                    f"- livebait: rate_ratio = {rr:.3f} (95% CI {rr_ci_lo:.3f}–{rr_ci_hi:.3f}), p = {pval:.3g}. "
                    f"Values >1 indicate higher fish/hour when using live bait. "
                )
            elif var == 'camper':
                desc_lines.append(
                    f"- camper: rate_ratio = {rr:.3f} (95% CI {rr_ci_lo:.3f}–{rr_ci_hi:.3f}), p = {pval:.3g}. "
                    f"Values >1 indicate higher fish/hour for groups with a camper."
                )
            elif var == 'total_people':
                desc_lines.append(
                    f"- total_people: rate_ratio = {rr:.3f} (95% CI {rr_ci_lo:.3f}–{rr_ci_hi:.3f}), p = {pval:.3g}. "
                    f"This is the multiplicative change in fish/hour for each additional person in the group."
                )

    description = " ".join(desc_lines)

    return {"object": summary, "description": description}