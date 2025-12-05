def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels GLM (Poisson or Negative Binomial)
    to answer how using live bait affects the catch rate (fish per hour).

    Returns a dictionary:
      {
        "object": { ... numeric results ... },
        "description": "Interpretation text ..."
      }

    The "object" is a JSON-serializable dict containing:
      - model_family: str
      - dispersion: float (Pearson chi2 / df_resid if available)
      - coefficients: dict of {param: float}
      - pvalues: dict of {param: float}
      - conf_int: dict of {param: [ci_lower, ci_upper]}
      - irr: dict of {param: incidence rate ratio = exp(coef)}
      - irr_conf_int: dict of {param: [exp(ci_lower), exp(ci_upper)]}
      - livebait_focus: dict containing livebait-specific numbers (coef, p, ci, IRR, IRR_ci)
    """
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output does not look like a fitted statsmodels results object.")

    # Determine model family label if available
    model_family = getattr(res, 'model_fit_family', None)
    if model_family is None:
        # fall back to family attribute on model
        try:
            model_family = getattr(res.model.family, 'name', str(res.model.family))
        except Exception:
            model_family = 'unknown'

    # Extract coefficients, p-values, CI
    params = res.params.copy()
    # convert to plain dict of floats
    coef_dict = {str(k): float(v) for k, v in params.items()}

    # p-values
    try:
        pvals = res.pvalues
        pval_dict = {str(k): float(v) for k, v in pvals.items()}
    except Exception:
        pval_dict = {}

    # confidence intervals
    try:
        ci_df = res.conf_int()
        # conf_int may be a DataFrame or ndarray; use index/position mapping against params
        ci_dict = {}
        for i, name in enumerate(params.index):
            try:
                lower = float(ci_df.iloc[i, 0])
                upper = float(ci_df.iloc[i, 1])
            except Exception:
                # fallback by name lookup
                try:
                    lower = float(ci_df.loc[name].iloc[0])
                    upper = float(ci_df.loc[name].iloc[1])
                except Exception:
                    lower = None
                    upper = None
            ci_dict[str(name)] = [lower, upper]
    except Exception:
        ci_dict = {}

    # Incidence rate ratios (IRR) = exp(coef) and IRR CIs
    irr_dict = {}
    irr_ci_dict = {}
    for name, coef in coef_dict.items():
        try:
            irr = float(np.exp(coef))
        except Exception:
            irr = None
        irr_dict[name] = irr
        ci = ci_dict.get(name)
        if ci and (ci[0] is not None) and (ci[1] is not None):
            try:
                irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
            except Exception:
                irr_ci = [None, None]
        else:
            irr_ci = [None, None]
        irr_ci_dict[name] = irr_ci

    # Compute Pearson dispersion statistic if not attached
    dispersion = getattr(res, 'dispersion_stat', None)
    if dispersion is None:
        try:
            rp = res.resid_pearson
            pearson_chi2 = float(np.sum(np.asarray(rp) ** 2))
            df_resid = float(res.df_resid) if hasattr(res, 'df_resid') else max(1.0, len(res.model.endog) - len(res.params))
            dispersion = pearson_chi2 / max(df_resid, 1.0)
        except Exception:
            dispersion = None

    # Focus on livebait variable (primary predictor)
    # Try exact name, then case-insensitive search
    livebait_name = None
    for name in coef_dict.keys():
        if name == 'livebait':
            livebait_name = name
            break
    if livebait_name is None:
        for name in coef_dict.keys():
            if str(name).lower() == 'livebait':
                livebait_name = name
                break

    livebait_focus = {}
    if livebait_name is not None:
        lb_coef = coef_dict[livebait_name]
        lb_p = pval_dict.get(livebait_name, None)
        lb_ci = ci_dict.get(livebait_name, [None, None])
        lb_irr = irr_dict.get(livebait_name, None)
        lb_irr_ci = irr_ci_dict.get(livebait_name, [None, None])

        # Simple significance statement
        significance = None
        try:
            if lb_p is not None:
                significance = "statistically significant (p < 0.05)" if lb_p < 0.05 else "not statistically significant (p >= 0.05)"
        except Exception:
            significance = None

        livebait_focus = {
            "param_name": str(livebait_name),
            "coef": lb_coef,
            "pvalue": lb_p,
            "conf_int_coef": lb_ci,
            "irr": lb_irr,
            "conf_int_irr": lb_irr_ci,
            "significance_statement": significance
        }
    else:
        livebait_focus = {
            "error": "Variable 'livebait' not found among fitted model parameters.",
            "available_params": list(coef_dict.keys())
        }

    # Package results
    result_object = {
        "model_family": model_family,
        "dispersion": float(dispersion) if dispersion is not None else None,
        "coefficients": coef_dict,
        "pvalues": pval_dict,
        "conf_int": ci_dict,
        "irr": irr_dict,
        "irr_conf_int": irr_ci_dict,
        "livebait_focus": livebait_focus
    }

    # Compose human-readable description
    if livebait_name is not None:
        desc_lines = []
        desc_lines.append(
            "Modeling fish_caught with log(hours) as an offset means coefficients are log(rate per hour)."
        )
        lb = livebait_focus
        desc_lines.append(
            f"The estimated coefficient for '{lb['param_name']}' is {lb['coef']:.4f} "
            f"(p = {lb['pvalue']:.4g}) with 95% CI on the coefficient {lb['conf_int_coef']}."
        )
        desc_lines.append(
            f"Exponentiating gives an incidence rate ratio (IRR) = {lb['irr']:.4f} "
            f"with 95% CI {lb['conf_int_irr']}."
        )
        if lb.get('significance_statement'):
            desc_lines.append(f"Interpretation: groups using live bait have a catch rate per hour that is multiplied by the IRR compared to groups not using live bait; this effect is {lb['significance_statement']}.")
        else:
            desc_lines.append("Interpretation: groups using live bait have a catch rate per hour multiplied by the IRR compared to groups not using live bait.")
        if result_object["dispersion"] is not None:
            desc_lines.append(f"Dispersion statistic (Pearson chi2 / df_resid) = {result_object['dispersion']:.3f}; values >>1 indicate overdispersion.")
        description = " ".join(desc_lines)
    else:
        description = ("Could not find a parameter named 'livebait' in the fitted model. "
                       "Returned available parameter estimates in 'object'.")
    return {"object": result_object, "description": description}