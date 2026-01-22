def extract_final_answer(model_output):
    """
    Extracts the effect of 'HasChildren' from a fitted model object (ZINB, NB GLM, or OLS fallback).
    Returns a dictionary with keys:
      - "object": dict of numeric results (coefficients, p-values, CIs, IRR/odds, significance flags)
      - "description": short human-readable interpretation of the results in context.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to safely get attributes from the results object
    def _get_series(attr_name):
        try:
            series = getattr(res, attr_name)
            # If it's a numpy array with index available separately
            if isinstance(series, (pd.Series, pd.DataFrame)):
                return series
            # try convert to pandas Series if possible using parameter names
            params = getattr(res, "params", None)
            if params is not None:
                # if params is a Series, use its index
                if isinstance(params, pd.Series):
                    return pd.Series(series, index=params.index)
            # Otherwise return None
            return None
        except Exception:
            return None

    params = _get_series("params")
    pvalues = _get_series("pvalues")
    confint = None
    try:
        ci = res.conf_int()
        # make sure it's a DataFrame
        if isinstance(ci, (list, tuple, np.ndarray)):
            # unlikely, but guard
            confint = pd.DataFrame(ci)
        else:
            confint = ci
    except Exception:
        confint = None

    # Prepare default output
    out = {
        "count_coef": None,
        "count_pvalue": None,
        "count_ci_lower": None,
        "count_ci_upper": None,
        "IRR": None,
        "percent_change": None,
        "count_significant": None,
        "inflation_exists": False,
        "inflation_coef": None,
        "inflation_pvalue": None,
        "inflation_ci_lower": None,
        "inflation_ci_upper": None,
        "inflation_odds_ratio": None,
        "inflation_significant": None,
        "notes": []
    }

    if params is None:
        return {
            "object": out,
            "description": "Could not find parameter estimates on the provided model object."
        }

    # Find count-part parameter name for HasChildren
    # Commonly it's exactly 'HasChildren' for count models.
    param_index = list(params.index)
    # Direct match
    count_name = None
    if "HasChildren" in param_index:
        count_name = "HasChildren"
    else:
        # Try to find a parameter that ends/contains HasChildren but is not an inflate param
        matches = [n for n in param_index if "HasChildren" in n and "inflate" not in n and "infl" not in n]
        if matches:
            count_name = matches[0]

    if count_name is None:
        out["notes"].append("No count-part parameter matching 'HasChildren' found in model parameters.")
    else:
        # extract numeric values
        try:
            coef = float(params[count_name])
            out["count_coef"] = coef
        except Exception:
            out["notes"].append(f"Unable to convert count coefficient '{count_name}' to float.")
            coef = None

        # p-value
        if pvalues is not None and count_name in pvalues.index:
            try:
                pval = float(pvalues[count_name])
                out["count_pvalue"] = pval
                out["count_significant"] = bool(pval < 0.05)
            except Exception:
                out["notes"].append("Unable to extract count p-value.")
        else:
            out["notes"].append("Count-part p-value not available in model output.")

        # confidence interval
        if confint is not None and count_name in confint.index:
            try:
                ci_low, ci_high = confint.loc[count_name].tolist()
                out["count_ci_lower"] = float(ci_low)
                out["count_ci_upper"] = float(ci_high)
            except Exception:
                out["notes"].append("Unable to extract count confidence interval.")
        else:
            out["notes"].append("Count-part confidence interval not available.")

        # IRR and percent change (for count model coefficient interpreted on log scale)
        if coef is not None:
            try:
                irr = float(np.exp(coef))
                out["IRR"] = irr
                out["percent_change"] = (irr - 1.0) * 100.0
            except Exception:
                out["notes"].append("Unable to compute IRR / percent change for count coefficient.")

    # Now attempt to find inflation-part parameter for HasChildren (if present)
    # Typical names: 'inflate_HasChildren' or 'inflate.HasChildren'
    infl_name = None
    # common patterns
    candidates = [n for n in param_index if "HasChildren" in n and ("inflate" in n or "infl" in n)]
    if candidates:
        infl_name = candidates[0]
    else:
        # Sometimes inflation params have prefix 'inflate_' exactly
        alt = "inflate_HasChildren"
        if alt in param_index:
            infl_name = alt

    if infl_name is not None:
        out["inflation_exists"] = True
        try:
            infl_coef = float(params[infl_name])
            out["inflation_coef"] = infl_coef
        except Exception:
            out["notes"].append(f"Unable to convert inflation coefficient '{infl_name}' to float.")
            infl_coef = None

        if pvalues is not None and infl_name in pvalues.index:
            try:
                infl_p = float(pvalues[infl_name])
                out["inflation_pvalue"] = infl_p
                out["inflation_significant"] = bool(infl_p < 0.05)
            except Exception:
                out["notes"].append("Unable to extract inflation p-value.")
        else:
            out["notes"].append("Inflation-part p-value not available in model output.")

        if confint is not None and infl_name in confint.index:
            try:
                infl_low, infl_high = confint.loc[infl_name].tolist()
                out["inflation_ci_lower"] = float(infl_low)
                out["inflation_ci_upper"] = float(infl_high)
            except Exception:
                out["notes"].append("Unable to extract inflation confidence interval.")
        else:
            out["notes"].append("Inflation-part confidence interval not available.")

        # For inflation (logit), exp(coef) gives odds ratio for being in the "always zero" group
        if infl_coef is not None:
            try:
                infl_or = float(np.exp(infl_coef))
                out["inflation_odds_ratio"] = infl_or
            except Exception:
                out["notes"].append("Unable to compute inflation odds ratio.")
    else:
        out["notes"].append("No inflation-part parameter for 'HasChildren' found; model may not be zero-inflated or uses different naming.")

    # Construct a short human-readable description/interpretation
    desc_parts = []
    if out["count_coef"] is not None:
        desc_parts.append(
            f"Count model: coefficient for HasChildren = {out['count_coef']:.4f} "
            f"(p = {out['count_pvalue']:.4f})" if out["count_pvalue"] is not None else
            f"Count model: coefficient for HasChildren = {out['count_coef']:.4f} (p unavailable)"
        )
        if out["IRR"] is not None:
            desc_parts.append(
                f"This corresponds to an incidence rate ratio (IRR) = {out['IRR']:.3f}, "
                f"i.e. expected affairs change of {out['percent_change']:.1f}% among the 'count' (susceptible) group."
            )
        if out["count_significant"] is True:
            dir_text = "decrease" if out["IRR"] < 1 else "increase"
            desc_parts.append(f"The effect is statistically significant (p < 0.05), indicating a {dir_text} in expected affair counts.")
        elif out["count_significant"] is False:
            desc_parts.append("The effect is not statistically significant (p >= 0.05) in the count part.")

    if out["inflation_exists"]:
        if out["inflation_coef"] is not None:
            desc_parts.append(
                f"Inflation model (structural-zero logit): coefficient for HasChildren = {out['inflation_coef']:.4f} "
                f"(p = {out['inflation_pvalue']:.4f})" if out["inflation_pvalue"] is not None else
                f"Inflation model: coefficient for HasChildren = {out['inflation_coef']:.4f} (p unavailable)"
            )
            # Interpret inflation sign: positive -> greater log-odds of being always-zero (i.e., less likely to have any affair)
            if out["inflation_odds_ratio"] is not None:
                or_text = f"odds ratio = {out['inflation_odds_ratio']:.3f}"
            else:
                or_text = "odds ratio unavailable"
            desc_parts.append(f"A positive inflation coefficient means higher odds of being in the 'always-zero' group (i.e., less likely to have any affair). {or_text}.")
            if out["inflation_significant"] is True:
                sign_dir = "increases" if out["inflation_coef"] > 0 else "decreases"
                desc_parts.append(f"This inflation effect is statistically significant (p < 0.05) and therefore {sign_dir} the probability of reporting zero affairs.")
            elif out["inflation_significant"] is False:
                desc_parts.append("The inflation effect is not statistically significant (p >= 0.05).")

    # Combined concluding sentence:
    # Determine overall sign/evidence that having children decreases engagement in affairs.
    conclusion = None
    # Heuristic: if count IRR <1 and significant OR inflation coef positive and significant -> evidence children decrease affairs.
    evidence_decrease = False
    evidence_increase = False
    if out["count_significant"] is True and out["IRR"] is not None:
        if out["IRR"] < 1:
            evidence_decrease = True
        elif out["IRR"] > 1:
            evidence_increase = True
    if out["inflation_significant"] is True and out["inflation_coef"] is not None:
        if out["inflation_coef"] > 0:
            evidence_decrease = True
        elif out["inflation_coef"] < 0:
            evidence_increase = True

    if evidence_decrease and not evidence_increase:
        conclusion = "Overall interpretation: Statistical evidence that having children is associated with LOWER engagement in extramarital affairs (either via lower expected counts or higher probability of being a structural zero)."
    elif evidence_increase and not evidence_decrease:
        conclusion = "Overall interpretation: Statistical evidence that having children is associated with HIGHER engagement in extramarital affairs."
    elif (evidence_decrease and evidence_increase) or (evidence_decrease is False and evidence_increase is False):
        # conflicting or no clear significant evidence
        conclusion = "Overall interpretation: No clear, consistent statistically significant evidence that having children decreases (or increases) engagement in extramarital affairs based on the extracted coefficients and significance tests."

    desc_parts.append(conclusion)
    description = " ".join([p for p in desc_parts if p is not None])

    return {
        "object": out,
        "description": description
    }