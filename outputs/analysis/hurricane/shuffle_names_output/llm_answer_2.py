def extract_final_answer(model_output):
    """
    Extract statistics for the 'femininity_z' coefficient from a fitted statsmodels GLMResultsWrapper
    (Negative Binomial) and return a concise summary.

    Returns a dict with:
      - "object": a dict of numeric values (coef, SE, p-value, z/t value, 95% CI, IRR and IRR CI, nobs)
      - "description": a short textual interpretation in the context of the hypothesis
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic checks and retrieval
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()  # DataFrame with two columns (lower, upper)
    except Exception as e:
        raise ValueError(f"Unrecognized model output object or missing attributes: {e}")

    if 'femininity_z' not in params.index:
        raise KeyError("The fitted model does not contain a coefficient named 'femininity_z'")

    # Extract stats
    coef = float(params.loc['femininity_z'])
    se = float(bse.loc['femininity_z']) if ('femininity_z' in getattr(bse, 'index', [])) else None
    pval = float(pvalues.loc['femininity_z'])
    ci_row = conf.loc['femininity_z']
    ci_lower = float(ci_row.iloc[0])
    ci_upper = float(ci_row.iloc[1])

    # z/t value (GLM typically uses z-values)
    z_or_t = None
    if hasattr(res, 'tvalues') and 'femininity_z' in getattr(res, 'tvalues', pd.Series()).index:
        try:
            z_or_t = float(res.tvalues.loc['femininity_z'])
        except Exception:
            z_or_t = None
    elif hasattr(res, 'zvalues') and 'femininity_z' in getattr(res, 'zvalues', pd.Series()).index:
        try:
            z_or_t = float(res.zvalues.loc['femininity_z'])
        except Exception:
            z_or_t = None

    # Incident rate ratio and its CI (useful interpretation for count models)
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Number of observations if available
    nobs = None
    if hasattr(res, 'nobs'):
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None
    elif hasattr(res, 'model') and hasattr(res.model, 'endog'):
        try:
            endog = res.model.endog
            if hasattr(endog, 'shape'):
                nobs = int(endog.shape[0])
            else:
                # fallback if endog is a 1d array-like without shape
                nobs = int(len(endog))
        except Exception:
            nobs = None

    # Quick interpretation relative to hypothesis
    signif = (pval < 0.05)
    if signif:
        if coef > 0:
            conclusion = ("Statistically significant positive association: higher femininity rating is associated "
                          "with more deaths (consistent with the hypothesis that more feminine names lead to fewer "
                          "precautions and thus higher fatalities).")
        else:
            conclusion = ("Statistically significant negative association: higher femininity rating is associated "
                          "with fewer deaths (contrary to the hypothesis).")
    else:
        conclusion = ("No statistically significant association between name femininity and deaths (p >= 0.05); "
                      "the data do not provide evidence supporting the hypothesis in this archival test.")

    result_object = {
        "coef": coef,
        "std_err": se,
        "p_value": pval,
        "z_or_t": z_or_t,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "incidence_rate_ratio": irr,
        "irr_ci_95_lower": irr_ci_lower,
        "irr_ci_95_upper": irr_ci_upper,
        "nobs": nobs
    }

    # Format numeric components safely for the description string
    se_str = f"{se:.4f}" if se is not None else "NA"
    zt_str = f"{z_or_t:.4f}" if z_or_t is not None else "NA"

    description = (
        f"Extracted 'femininity_z' coefficient from Negative Binomial GLM: coef={coef:.4f}, "
        f"SE={se_str}, z/t={zt_str}, "
        f"p={pval:.3g}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f}). IRR=exp(coef)={irr:.3f} "
        f"(95% CI: {irr_ci_lower:.3f}–{irr_ci_upper:.3f}). {conclusion}"
    )

    return {"object": result_object, "description": description}