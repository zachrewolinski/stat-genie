def extract_final_answer(model_output):
    """
    Extract statistics related to the effect of HasChildren on AffairsCount from
    a fitted statsmodels ZeroInflatedPoissonResultsWrapper.
    Returns a dict with keys:
      - "object": dict with numeric results (count and inflation parts)
      - "description": brief interpretation in plain language
    """
    import numpy as np
    import pandas as pd

    res = model_output  # expected to be a statsmodels results wrapper

    # Basic parameter table
    try:
        params = res.params.copy()           # pandas Series
        bse = res.bse.copy()                 # pandas Series
        pvals = res.pvalues.copy()           # pandas Series
        conf = res.conf_int().copy()         # DataFrame with columns [0,1] or named
    except Exception as e:
        raise ValueError(f"Provided object does not look like a statsmodels results wrapper: {e}")

    # Helper to extract stats for a given parameter name
    def _get_stats(param_name):
        if param_name not in params.index:
            return None
        coef = float(params.loc[param_name])
        se = float(bse.loc[param_name]) if param_name in bse.index else None
        pval = float(pvals.loc[param_name]) if param_name in pvals.index else None
        # z-stat if se available
        z = float(coef / se) if se not in (None, 0) else None
        # confidence interval
        if param_name in conf.index:
            ci_lower = float(conf.loc[param_name].iloc[0])
            ci_upper = float(conf.loc[param_name].iloc[1])
        else:
            ci_lower = ci_upper = None
        return {
            "coef": coef,
            "se": se,
            "z": z,
            "pval": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    # Count model parameter for HasChildren (expected name: 'HasChildren')
    count_stats = _get_stats('HasChildren')

    # Inflation model parameter for HasChildren (likely named 'inflate_HasChildren')
    infl_stats = _get_stats('inflate_HasChildren')

    # If inflation param uses different naming convention, try prefix 'inflate.' or suffix
    if infl_stats is None:
        # look for any param name containing 'inflate' and 'HasChildren'
        match = [n for n in params.index if ('inflate' in n.lower() or 'infl.' in n.lower()) and 'haschildren'.lower() in n.lower()]
        if match:
            infl_stats = _get_stats(match[0])

    # Compute IRR (incidence rate ratio) and its CI for the count coefficient
    if count_stats is not None:
        irr = float(np.exp(count_stats["coef"]))
        if count_stats["ci_lower"] is not None:
            irr_ci_lower = float(np.exp(count_stats["ci_lower"]))
            irr_ci_upper = float(np.exp(count_stats["ci_upper"]))
        else:
            irr_ci_lower = irr_ci_upper = None
        count_stats.update({"IRR": irr, "IRR_ci_lower": irr_ci_lower, "IRR_ci_upper": irr_ci_upper})
    else:
        # If missing, provide an informative None
        count_stats = None

    # For inflation, interpret coefficient on logit scale: positive coef -> higher odds of being an "excess zero" (i.e., structural non-affair)
    if infl_stats is not None:
        infl_odds_ratio = float(np.exp(infl_stats["coef"]))
        if infl_stats["ci_lower"] is not None:
            infl_or_ci_lower = float(np.exp(infl_stats["ci_lower"]))
            infl_or_ci_upper = float(np.exp(infl_stats["ci_upper"]))
        else:
            infl_or_ci_lower = infl_or_ci_upper = None
        infl_stats.update({"odds_ratio": infl_odds_ratio, "odds_ratio_ci_lower": infl_or_ci_lower, "odds_ratio_ci_upper": infl_or_ci_upper})

    # Formulate a concise conclusion about whether having children decreases extramarital affairs.
    conclusion_parts = []
    if count_stats is None:
        conclusion_parts.append("Count-model coefficient for HasChildren not found in the results.")
    else:
        p = count_stats["pval"]
        coef = count_stats["coef"]
        irr = count_stats.get("IRR", None)
        if p is None:
            conclusion_parts.append("No p-value available for the count-model HasChildren coefficient.")
        else:
            if p < 0.05:
                if coef < 0:
                    conclusion_parts.append(
                        f"Statistically significant evidence (p = {p:.3g}) that having children is associated with fewer reported affairs: "
                        f"count coef = {coef:.4f}, IRR = {irr:.3f} (95% CI [{count_stats['IRR_ci_lower']:.3f}, {count_stats['IRR_ci_upper']:.3f}])."
                    )
                else:
                    conclusion_parts.append(
                        f"Statistically significant evidence (p = {p:.3g}) that having children is associated with more reported affairs: "
                        f"count coef = {coef:.4f}, IRR = {irr:.3f}."
                    )
            elif p < 0.10:
                if coef < 0:
                    conclusion_parts.append(
                        f"Weak evidence (p = {p:.3g}) that having children is associated with fewer affairs (coef = {coef:.4f}, IRR = {irr:.3f})."
                    )
                else:
                    conclusion_parts.append(
                        f"Weak evidence (p = {p:.3g}) that having children is associated with more affairs (coef = {coef:.4f}, IRR = {irr:.3f})."
                    )
            else:
                conclusion_parts.append(
                    f"No statistically significant association between having children and the expected number of affairs (coef = {coef:.4f}, p = {p:.3g}, IRR = {irr:.3f})."
                )

    # Add inflation interpretation if available
    if infl_stats is not None:
        p_inf = infl_stats["pval"]
        coef_inf = infl_stats["coef"]
        or_inf = infl_stats.get("odds_ratio", None)
        if p_inf is not None and p_inf < 0.05:
            if coef_inf > 0:
                conclusion_parts.append(
                    f"Additionally, the inflation part shows a significant positive association (p = {p_inf:.3g}) for HasChildren "
                    f"(inflate coef = {coef_inf:.4f}, odds ratio = {or_inf:.3f}), meaning those with children are more likely to be in the "
                    "structural-zero group (i.e., more likely to report zero affairs). This supports the interpretation that children are associated with fewer affairs."
                )
            else:
                conclusion_parts.append(
                    f"Inflation part shows a significant negative association (p = {p_inf:.3g}) for HasChildren "
                    f"(inflate coef = {coef_inf:.4f}, odds ratio = {or_inf:.3f}), meaning those with children are less likely to be in the structural-zero group."
                )
        else:
            conclusion_parts.append("No statistically significant effect of HasChildren in the zero-inflation component was detected.")

    conclusion_text = " ".join(conclusion_parts)

    # Build the object to return: numeric results plus an overall short verdict
    result_object = {
        "count_model_HasChildren": count_stats,
        "inflation_model_HasChildren": infl_stats,
        "verdict": None
    }

    # Short verdict based primarily on count-model p-value and sign
    if count_stats is None or count_stats.get("pval") is None:
        result_object["verdict"] = "Insufficient information to determine effect of HasChildren from the fitted model output."
    else:
        p = count_stats["pval"]
        coef = count_stats["coef"]
        if p < 0.05 and coef < 0:
            result_object["verdict"] = "Having children is associated with a statistically significant decrease in reported affairs."
        elif p < 0.05 and coef > 0:
            result_object["verdict"] = "Having children is associated with a statistically significant increase in reported affairs."
        elif p < 0.10 and coef < 0:
            result_object["verdict"] = "Weak evidence that having children decreases reported affairs (p < 0.10)."
        else:
            result_object["verdict"] = "No statistically significant evidence that having children affects reported affairs."

    return {
        "object": result_object,
        "description": (
            "Extracted coefficient, standard error, z, p-value, and 95% CI for the 'HasChildren' parameter "
            "in both the count and zero-inflation parts of the ZIP model (when present). "
            "The count-model coefficient (log link) -> exponentiated to get IRR: IRR < 1 implies fewer expected affairs for those with children. "
            "The inflation-model coefficient (logit link) -> exponentiated to get odds ratio: positive inflation coef (OR>1) implies higher probability of being a structural zero (i.e., more likely to report zero affairs). "
            "The 'verdict' field gives a concise conclusion based primarily on the count-model p-value and sign; the full numeric results are available under 'object'. "
            "Interpretation: combine count-model IRR and inflation-model sign to assess whether having children is associated with fewer extramarital affairs."
        )
    }