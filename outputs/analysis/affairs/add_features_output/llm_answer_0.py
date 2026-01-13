def extract_final_answer(model_output):
    """
    Extracts statistics relating to the effect of HasChildren from a fitted
    statsmodels ZeroInflatedNegativeBinomialResultsWrapper (as returned by the
    modeling function in the task). Returns a dict with a numeric 'object'
    (detailed measurements) and a short 'description' interpreting them.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    result_dict = {}
    # 1) Average marginal effect (computed by the modeling function, if present)
    ame = getattr(res, "ame_HasChildren", None)
    result_dict["ame_HasChildren"] = (None if ame is None else float(ame))

    # Helper to extract a parameter by name (label or by searching model.exog_names)
    def _extract_param(name):
        try:
            params = res.params  # pandas Series
            bse = res.bse
            pvals = res.pvalues
            ci = res.conf_int()
        except Exception:
            return None

        # Try by label first
        if name in params.index:
            return {
                "coef": float(params[name]),
                "se": float(bse[name]) if name in bse.index else None,
                "pvalue": float(pvals[name]) if name in pvals.index else None,
                "ci_lower": float(ci.loc[name][0]) if name in ci.index else None,
                "ci_upper": float(ci.loc[name][1]) if name in ci.index else None,
            }

        # If not found by label, try to locate by position using model.exog_names
        try:
            exog_names = getattr(res.model, "exog_names", None)
            if exog_names and name in exog_names:
                pos = exog_names.index(name)
                # params can be indexed positionally
                return {
                    "coef": float(params.iloc[pos]),
                    "se": float(bse.iloc[pos]) if bse is not None else None,
                    "pvalue": float(pvals.iloc[pos]) if pvals is not None else None,
                    "ci_lower": float(ci.iloc[pos, 0]) if ci is not None else None,
                    "ci_upper": float(ci.iloc[pos, 1]) if ci is not None else None,
                }
        except Exception:
            pass

        return None

    # Extract count-model parameter for HasChildren
    count_info = _extract_param("HasChildren")
    result_dict["count_param_HasChildren"] = count_info

    # Extract inflation-model parameter for HasChildren (common naming: 'inflate_<name>')
    infl_names_tried = []
    infl_info = None
    # Common statsmodels prefix for inflation params is 'inflate_' but try variations
    candidates = [f"inflate_HasChildren", "inflate.HasChildren", "HasChildren_infl", "HasChildren_inflation", "HasChildren"]
    # But prefer explicit inflation names from model if available
    try:
        exog_infl_names = getattr(res.model, "exog_infl_names", None)
        if exog_infl_names:
            # If we can find one that matches, build expected label used by params index
            # Statsmodels typically uses 'inflate_' + name in params index; try that
            for nm in exog_infl_names:
                if nm == "HasChildren" or "HasChildren" in nm:
                    infl_names_tried.append(f"inflate_{nm}")
                    infl_names_tried.append(nm)
    except Exception:
        pass
    infl_names_tried.extend(candidates)
    # Deduplicate while preserving order
    seen = set()
    infl_names_tried = [x for x in infl_names_tried if not (x in seen or seen.add(x))]

    for n in infl_names_tried:
        infl_info = _extract_param(n)
        if infl_info is not None:
            # Mark which label matched
            infl_info["_matched_name"] = n
            break

    result_dict["inflation_param_HasChildren"] = infl_info

    # Compute IRR (incidence rate ratio) and its CI for the count coefficient, if available
    if count_info is not None and count_info.get("coef") is not None:
        coef = count_info["coef"]
        ci_lo = count_info["ci_lower"]
        ci_hi = count_info["ci_upper"]
        irr = float(np.exp(coef))
        irr_ci_lo = (float(np.exp(ci_lo)) if ci_lo is not None else None)
        irr_ci_hi = (float(np.exp(ci_hi)) if ci_hi is not None else None)
        result_dict["count_param_HasChildren_IRR"] = {
            "irr": irr,
            "irr_ci_lower": irr_ci_lo,
            "irr_ci_upper": irr_ci_hi,
        }
    else:
        result_dict["count_param_HasChildren_IRR"] = None

    # Short interpretive description
    desc_lines = []
    # AME interpretation
    if result_dict["ame_HasChildren"] is not None:
        ame_val = result_dict["ame_HasChildren"]
        desc_lines.append(
            f"Average marginal effect (HasChildren: 0 -> 1) on expected number of affairs = {ame_val:.4f}.\n"
            "This is the model-computed mean change in expected affairs when changing HasChildren from 0 to 1,\n"
            "holding other covariates at their observed values (positive => more affairs with children; negative => fewer)."
        )
    else:
        desc_lines.append("Average marginal effect for HasChildren not available from the model object.")

    # Count model interpretation
    if count_info is not None:
        coef = count_info["coef"]
        pval = count_info["pvalue"]
        se = count_info["se"]
        ci_lo = count_info["ci_lower"]
        ci_hi = count_info["ci_upper"]
        irr_block = result_dict["count_param_HasChildren_IRR"]
        sig = (pval is not None and pval < 0.05)
        desc_lines.append(
            f"Count model (negative binomial) coefficient for HasChildren = {coef:.4f} "
            f"(SE = {se:.4f}, p = {pval:.4g}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}])."
        )
        if irr_block:
            desc_lines.append(
                f"Incidence Rate Ratio (exp(coef)) = {irr_block['irr']:.4f} "
                f"(95% CI = [{irr_block['irr_ci_lower']:.4f}, {irr_block['irr_ci_upper']:.4f}])."
            )
        if sig:
            # direction
            direction = "decrease" if coef < 0 else "increase"
            desc_lines.append(f"Interpretation: Having children is associated with a statistically significant {direction} in the expected count of affairs (per the count part, p < .05).")
        else:
            desc_lines.append("Interpretation: The count-model coefficient is not statistically significant at conventional levels (p >= 0.05).")
    else:
        desc_lines.append("Count-model parameter for HasChildren could not be located in the model output.")

    # Inflation model interpretation
    if infl_info is not None:
        coef = infl_info.get("coef")
        pval = infl_info.get("pvalue")
        se = infl_info.get("se")
        ci_lo = infl_info.get("ci_lower")
        ci_hi = infl_info.get("ci_upper")
        matched = infl_info.get("_matched_name", "unknown")
        desc_lines.append(
            f"Inflation model parameter matched as '{matched}': coef = {coef:.4f} (SE = {se:.4f}, p = {pval:.4g}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}])."
        )
        desc_lines.append(
            "Interpretation: In the zero-inflation (logit) part, a positive coefficient means higher log-odds of being an 'excess-zero' (i.e., structural zero)."
        )
    else:
        desc_lines.append("Inflation-model parameter for HasChildren was not clearly identified in the model output (no inflation effect extracted).")

    # Final concise conclusion
    # Use AME if available for plain-language effect on expected count; otherwise use sign/pval of count coef.
    conclusion = ""
    if result_dict["ame_HasChildren"] is not None:
        ame_val = result_dict["ame_HasChildren"]
        if count_info is not None and count_info.get("pvalue") is not None and count_info["pvalue"] < 0.05:
            sig_text = "statistically significant"
        else:
            sig_text = "not statistically significant"
        if ame_val < 0:
            conclusion = f"Overall: Having children is associated with a decrease in expected number of affairs by about {abs(ame_val):.4f} affairs on average; this effect is {sig_text} (see count-model p-value)."
        elif ame_val > 0:
            conclusion = f"Overall: Having children is associated with an increase in expected number of affairs by about {ame_val:.4f} affairs on average; this effect is {sig_text}."
        else:
            conclusion = "Overall: No average effect of having children on expected number of affairs (AME = 0)."
    else:
        # fallback: use count coefficient sign
        if count_info is not None:
            coef = count_info["coef"]
            pval = count_info["pvalue"]
            sig_text = "statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant"
            if coef < 0:
                conclusion = f"Overall: The count-model coefficient suggests having children is associated with fewer affairs (coef {coef:.4f}); this effect is {sig_text}."
            elif coef > 0:
                conclusion = f"Overall: The count-model coefficient suggests having children is associated with more affairs (coef {coef:.4f}); this effect is {sig_text}."
            else:
                conclusion = "Overall: No effect detected for HasChildren in the count model."
        else:
            conclusion = "Overall: Unable to determine effect of HasChildren from the provided model object."

    desc_lines.append(conclusion)

    description = "\n\n".join(desc_lines)

    return {"object": result_dict, "description": description}