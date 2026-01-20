def extract_final_answer(model_output):
    """
    Extracts statistics for the primary independent variable(s) from a fitted statsmodels GLM results object
    (optionally cluster-robust results returned by get_robustcov_results).

    Returns a dictionary with keys:
      - "object": a dict containing extracted numeric results for 'is_dark' (and 'skin_rating' if present)
      - "description": a brief plain-language explanation of what the numbers mean regarding whether
                       darker-skinned players are more likely to receive red cards.

    The "object" dict structure:
      {
        "is_dark": {
          "coef": float,                # log rate ratio (coefficient from NB model)
          "se": float,                  # standard error
          "p_value": float,
          "ci_lower": float,            # 95% CI on coef
          "ci_upper": float,
          "irr": float,                 # incidence rate ratio = exp(coef)
          "irr_ci_lower": float,
          "irr_ci_upper": float,
          "significant_0.05": bool,
          "conclusion": str             # short conclusion for this parameter
        },
        "skin_rating": { ... } or None,
        "final_answer": "Yes" / "No" / "Inconclusive"  # summary answering the task question
      }
    """
    import numpy as np

    res = model_output

    # Basic attribute checks
    for attr in ("params", "bse", "pvalues", "conf_int"):
        if not hasattr(res, attr):
            raise ValueError(f"Provided model_output does not have required attribute '{attr}'")

    params = res.params           # pandas Series expected
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()        # DataFrame or array-like with index matching params.index

    def _extract(name):
        if name not in params.index:
            return None
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else None
        p = float(pvalues[name]) if name in pvalues.index else None

        # get confidence interval robustly
        try:
            ci_row = conf.loc[name]
        except Exception:
            # fallback: conf may be numpy array with same row order as params
            idx = list(params.index).index(name)
            ci_row = conf[idx]
        ci_arr = np.asarray(ci_row, dtype=float)
        ci_lower, ci_upper = float(ci_arr[0]), float(ci_arr[1])

        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower))
        irr_ci_upper = float(np.exp(ci_upper))

        significant = (p is not None) and (p < 0.05)
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        conclusion = (
            f"{'Statistically significant: ' if significant else ''}"
            f"Estimated red-card rate is {direction} for a one-unit increase in {name} "
            f"(IRR={irr:.3f}, 95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p={p:.3f})"
            if p is not None else
            f"Estimated red-card rate is {direction} (IRR={irr:.3f}, 95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}])."
        )

        return {
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "irr": irr,
            "irr_ci_lower": irr_ci_lower,
            "irr_ci_upper": irr_ci_upper,
            "significant_0.05": bool(significant),
            "conclusion": conclusion
        }

    is_dark_stats = _extract("is_dark")
    skin_rating_stats = _extract("skin_rating")

    # Form final summary conclusion focused on the task question about darker vs lighter skin tone.
    # Use primary binary indicator 'is_dark' if available, otherwise use continuous 'skin_rating'.
    if is_dark_stats is not None:
        if is_dark_stats["significant_0.05"] and is_dark_stats["coef"] > 0:
            final_answer = "Yes"
            final_statement = (
                "The binary indicator 'is_dark' has a positive, statistically significant association with red-card rate. "
                "This implies darker-skinned players are more likely to receive red cards (see IRR and p-value)."
            )
        elif is_dark_stats["significant_0.05"] and is_dark_stats["coef"] <= 0:
            final_answer = "No"
            final_statement = (
                "The binary indicator 'is_dark' has a statistically significant association in the direction of fewer red cards "
                "for darker-skinned players."
            )
        else:
            final_answer = "Inconclusive"
            final_statement = (
                "The binary indicator 'is_dark' does not show a statistically significant association with red-card rate at alpha=0.05. "
                "Therefore the evidence is inconclusive for a difference in red-card rates by skin tone."
            )
    elif skin_rating_stats is not None:
        # fall back to continuous rating
        if skin_rating_stats["significant_0.05"] and skin_rating_stats["coef"] > 0:
            final_answer = "Yes"
            final_statement = (
                "The continuous skin_rating has a positive, statistically significant association with red-card rate, "
                "suggesting darker-rated players receive more red cards."
            )
        elif skin_rating_stats["significant_0.05"] and skin_rating_stats["coef"] <= 0:
            final_answer = "No"
            final_statement = (
                "The continuous skin_rating has a statistically significant association in the negative direction, "
                "suggesting darker-rated players receive fewer red cards."
            )
        else:
            final_answer = "Inconclusive"
            final_statement = (
                "The continuous skin_rating does not show a statistically significant association with red-card rate at alpha=0.05."
            )
    else:
        final_answer = "Inconclusive"
        final_statement = "Neither 'is_dark' nor 'skin_rating' appear in the model output parameters."

    output_object = {
        "is_dark": is_dark_stats,
        "skin_rating": skin_rating_stats,
        "final_answer": final_answer
    }

    description = (
        "Extracted coefficient(s), standard error(s), p-value(s), 95% confidence interval(s), and incidence-rate ratio(s) "
        "for the model parameter(s) related to skin tone. The 'final_answer' gives a concise summary (Yes/No/Inconclusive) "
        "about whether darker-skinned players are more likely to receive red cards, based primarily on the binary 'is_dark' "
        "variable (falls back to 'skin_rating' if 'is_dark' is not present). See the returned 'object' for numeric details."
    )

    return {"object": output_object, "description": description}