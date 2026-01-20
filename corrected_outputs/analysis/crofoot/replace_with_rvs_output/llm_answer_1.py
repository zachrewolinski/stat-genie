def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals,
    and odds ratios for the focal predictors in the fitted GLM results object.
    
    Expected model_output: a statsmodels results object (e.g., the object
    returned by get_robustcov_results on a GLMResultsWrapper).
    
    Returns:
      {
        "object": {
          "terms": {
            "<term>": {
              "coef": float,
              "se": float,
              "pvalue": float,
              "ci_lower": float,
              "ci_upper": float,
              "odds_ratio": float,
              "or_ci_lower": float,
              "or_ci_upper": float,
              "significant": bool
            }, ...
          },
          "note": "cluster-robust SEs used if available"
        },
        "description": "Plain-language interpretation of effects in context"
      }
    """
    import numpy as np

    # Terms of interest
    terms = ['size_ratio_z', 'dist_adv_z', 'size_dist_interaction']
    out = {"terms": {}}
    note = []
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError(f"model_output does not appear to have .params attribute: {e}")

    # Try to obtain standard errors, p-values, and confidence intervals
    # These should exist on the robust results object; handle gracefully if missing.
    bse = None
    pvalues = None
    ci = None
    try:
        bse = model_output.bse
    except Exception:
        bse = None
    try:
        pvalues = model_output.pvalues
    except Exception:
        pvalues = None
    try:
        ci = model_output.conf_int()
    except Exception:
        ci = None

    # Determine index labels for params
    param_index = list(params.index)

    for term in terms:
        if term not in param_index:
            out["terms"][term] = {
                "error": f"Term '{term}' not found in model parameters."
            }
            continue

        coef = float(params[term])
        se = float(bse[term]) if (bse is not None and term in bse.index) else None
        pval = float(pvalues[term]) if (pvalues is not None and term in pvalues.index) else None

        if ci is not None:
            # conf_int may be a DataFrame or ndarray; try to access by label first
            try:
                # DataFrame-like
                ci_lower = float(ci.loc[term][0])
                ci_upper = float(ci.loc[term][1])
            except Exception:
                try:
                    # ndarray-like with same ordering as params
                    idx = param_index.index(term)
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
                except Exception:
                    ci_lower = None
                    ci_upper = None
        else:
            ci_lower = None
            ci_upper = None

        # Odds ratio and its CI (if coef exists)
        try:
            or_val = float(np.exp(coef))
        except Exception:
            or_val = None
        try:
            or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            or_ci_lower = None
            or_ci_upper = None

        significant = (pval is not None) and (pval < 0.05)

        out["terms"][term] = {
            "coef": coef,
            "se": float(se) if se is not None else None,
            "pvalue": float(pval) if pval is not None else None,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "odds_ratio": or_val,
            "or_ci_lower": or_ci_lower,
            "or_ci_upper": or_ci_upper,
            "significant": bool(significant)
        }

    # Add a short note about clustered SEs if object appears to be robust results
    # Many robust result objects have attribute 'cov_type' or were produced by get_robustcov_results
    ctxt = None
    try:
        ctxt = getattr(model_output, "cov_type", None)
        if ctxt is not None:
            note.append(f"cov_type = {ctxt}")
    except Exception:
        pass
    # Heuristic: if model_output was produced via get_robustcov_results it often has .cov_kwds or .cov_type
    if note:
        out["note"] = "; ".join(note)
    else:
        out["note"] = "Cluster-robust SEs may have been used (as returned object)."

    # Compose a plain-language description that programmatically interprets the signs and significance
    desc_lines = []
    desc_lines.append("Model predicts focal group win (1) vs loss (0) from relative group size (size_ratio_z),")
    desc_lines.append("contest location advantage (dist_adv_z), their interaction, and controls (n_focal_z, m_diff_z).")
    desc_lines.append("Below we summarize the estimated direction and statistical evidence for each focal term:")

    for term in terms:
        info = out["terms"].get(term, {})
        if "error" in info:
            desc_lines.append(f"- {term}: {info['error']}")
            continue
        coef = info["coef"]
        pval = info["pvalue"]
        sign = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        sig_text = f"statistically significant (p = {pval:.3g})" if (pval is not None and pval < 0.05) else (
            f"not statistically significant (p = {pval:.3g})" if pval is not None else "p-value unavailable"
        )
        or_str = f"odds ratio = {info['odds_ratio']:.3g}" if info["odds_ratio"] is not None else "odds ratio unavailable"
        desc_lines.append(f"- {term}: {sign} effect on log-odds of winning; {sig_text}; {or_str}.")

    # If interaction is significant, provide interpretation guidance
    inter_info = out["terms"].get('size_dist_interaction', {})
    if isinstance(inter_info, dict) and inter_info.get("significant"):
        coef_inter = inter_info["coef"]
        if coef_inter > 0:
            inter_interp = ("The positive interaction indicates that the beneficial effect of being relatively larger "
                            "on the probability of winning becomes stronger when the focal group has greater home-range advantage "
                            "(i.e., is nearer its home center than the other group).")
        else:
            inter_interp = ("The negative interaction indicates that the beneficial effect of being relatively larger "
                            "on the probability of winning is reduced (or reversed) when the focal group has greater home-range advantage.")
        desc_lines.append(inter_interp)
    else:
        desc_lines.append("Interaction term is not statistically significant, so there is no strong evidence that the effect of relative group size depends on contest location.")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}