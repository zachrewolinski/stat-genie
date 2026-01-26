def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, 95% CIs and incidence-rate-ratios (IRRs)
    for the primary predictors "DarkSkin" and "SkinTone" from a fitted statsmodels result
    (robust/clustered-results wrapper or standard results).
    
    Returns a dictionary with:
      - "object": dict keyed by predictor containing numeric results
      - "description": brief textual interpretation (direction, significance, IRR with CI)
    """
    import numpy as np
    res = model_output

    # Try to access typical statsmodels result attributes
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        ci = res.conf_int()  # DataFrame with two columns [lower, upper]
    except Exception as e:
        raise ValueError("Provided model_output does not expose expected statsmodels attributes: params, bse, pvalues, conf_int(). Error: %s" % str(e))

    predictors = ['DarkSkin', 'SkinTone']
    result_obj = {}

    for pred in predictors:
        if pred in params.index:
            coef = float(params[pred])
            se = float(bse[pred]) if pred in bse.index else None
            pval = float(pvalues[pred]) if pred in pvalues.index else None

            # conf_int() may return a DataFrame with numeric column labels 0 and 1
            try:
                ci_lower = float(ci.loc[pred, ci.columns[0]])
                ci_upper = float(ci.loc[pred, ci.columns[1]])
            except Exception:
                # fallback if ci is array-like
                try:
                    row = ci[pred]
                    ci_lower = float(row[0]); ci_upper = float(row[1])
                except Exception:
                    ci_lower = None; ci_upper = None

            irr = float(np.exp(coef)) if coef is not None else None
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

            result_obj[pred] = {
                'coef': coef,
                'std_err': se,
                'p_value': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'IRR': irr,
                'IRR_ci_lower': irr_ci_lower,
                'IRR_ci_upper': irr_ci_upper
            }
        else:
            result_obj[pred] = None

    # Build a concise description that a researcher can use to decide yes/no.
    descr_parts = []
    for pred in predictors:
        info = result_obj[pred]
        if info is None:
            descr_parts.append(f"{pred}: not present in model results.")
            continue
        p = info['p_value']
        coef = info['coef']
        irr = info['IRR']
        irr_lo = info['IRR_ci_lower']
        irr_hi = info['IRR_ci_upper']

        sig_text = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant"
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no change")
        descr_parts.append(
            f"{pred}: coef={coef:.4f}, p={p:.4g} ({sig_text}); "
            f"interpreted as IRR={irr:.3f} (95% CI [{irr_lo:.3f}, {irr_hi:.3f}]) → {direction} rate of red cards."
        )

    description = " ".join(descr_parts)

    return {"object": result_obj, "description": description}