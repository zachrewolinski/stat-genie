def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether modern humans (Homo sapiens)
    have higher AMTL compared to non-human primate genera from a fitted
    statsmodels GLMResultsWrapper object.

    Returns a dictionary with keys:
      - "object": dict containing parameter estimates, p-values, confidence intervals,
                  odds ratios, and a final conclusion (yes/no/inconclusive) plus text.
      - "description": brief human-readable interpretation of the returned object.

    The function handles three common cases:
      1) a binary indicator "is_human" was included in the model -> use it directly.
      2) a genus dummy for Homo_sapiens exists (e.g., "genus_Homo_sapiens") -> use it.
      3) Homo_sapiens was the omitted reference level (no Homo dummy) -> inspect
         the other genus coefficients (Pan/Pongo/Papio) which are relative to Homo.
    """
    import numpy as np
    import math

    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Extract parameter estimates, p-values, and CIs
    try:
        params = model_output.params.copy()
        pvals = model_output.pvalues.copy()
        ci_df = model_output.conf_int()
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract stats from model_output: {e}"
        }

    # Helper to build a summary dict for a parameter name
    def summarize_param(name):
        coef = float(params[name])
        pval = float(pvals.get(name, np.nan))
        ci_low, ci_high = tuple(ci_df.loc[name])
        or_est = math.exp(coef)
        or_ci = (math.exp(ci_low), math.exp(ci_high))
        return {
            "coef": coef,
            "p_value": pval,
            "conf_int_95": (ci_low, ci_high),
            "odds_ratio": or_est,
            "odds_ratio_CI_95": or_ci,
            "significant_0.05": bool(pval < 0.05)
        }

    param_names = list(params.index.astype(str))

    # 1) Prefer an explicit 'is_human' predictor if present
    if "is_human" in param_names:
        name = "is_human"
        summary = summarize_param(name)
        if summary["coef"] > 0 and summary["significant_0.05"]:
            conclusion = ("yes", "The 'is_human' coefficient is positive and statistically significant "
                                 "— modern humans have higher AMTL (higher log-odds / odds ratio > 1).")
        elif summary["coef"] > 0 and not summary["significant_0.05"]:
            conclusion = ("inconclusive", "The 'is_human' coefficient is positive but not statistically significant.")
        elif summary["coef"] < 0 and summary["significant_0.05"]:
            conclusion = ("no", "The 'is_human' coefficient is negative and statistically significant — humans have lower AMTL.")
        else:
            conclusion = ("inconclusive", "The 'is_human' coefficient is negative but not statistically significant.")
        return {
            "object": {
                "method_used": "is_human",
                "parameter": name,
                "summary": summary,
                "conclusion_code": conclusion[0],
                "conclusion_text": conclusion[1]
            },
            "description": "Used the explicit is_human indicator (1 = Homo sapiens). Coefficient is on the log-odds scale; odds ratio reported."
        }

    # 2) Look for a genus dummy that names Homo (case-insensitive match)
    homo_param = None
    for n in param_names:
        low = n.lower()
        if ("homo" in low) or ("sapiens" in low) or ("homo_sapiens" in low) or ("homo-sapiens" in low):
            # ensure it is a genus dummy (commonly prefixed by 'genus_' but accept others)
            homo_param = n
            break

    if homo_param is not None:
        summary = summarize_param(homo_param)
        # positive coef -> this genus (Homo) has higher log-odds than the model reference level
        if summary["coef"] > 0 and summary["significant_0.05"]:
            conclusion = ("yes", f"The coefficient for {homo_param} is positive and statistically significant — Homo sapiens have higher AMTL than the reference genus.")
        elif summary["coef"] > 0 and not summary["significant_0.05"]:
            conclusion = ("inconclusive", f"The coefficient for {homo_param} is positive but not statistically significant.")
        elif summary["coef"] < 0 and summary["significant_0.05"]:
            conclusion = ("no", f"The coefficient for {homo_param} is negative and statistically significant — Homo sapiens have lower AMTL than the reference genus.")
        else:
            conclusion = ("inconclusive", f"The coefficient for {homo_param} is negative but not statistically significant.")
        return {
            "object": {
                "method_used": "genus_dummy_direct",
                "parameter": homo_param,
                "summary": summary,
                "conclusion_code": conclusion[0],
                "conclusion_text": conclusion[1]
            },
            "description": "Used the genus dummy that directly indicates Homo (coefficient compares Homo to the reference genus). Coefficient is on the log-odds scale; odds ratio reported."
        }

    # 3) If Homo is the omitted baseline, inspect other genus coefficients which are relative to Homo
    # Find genus-related parameters (those starting with 'genus_' or containing 'genus' in name)
    genus_params = [n for n in param_names if n.startswith("genus_") or n.lower().startswith("genus")]
    # If none explicitly labeled, fallback to parameters that look like genus levels (Pan/Pongo/Papio)
    if not genus_params:
        candidates = []
        for n in param_names:
            ln = n.lower()
            if any(sub in ln for sub in ("pan", "pongo", "papio", "pan_", "pongo_", "papio_")):
                candidates.append(n)
        genus_params = candidates

    if genus_params:
        summaries = {n: summarize_param(n) for n in genus_params}
        # For each non-human genus param: coefficient = (that genus) - (reference genus)
        # If Homo is reference, a negative coef means the non-human genus has lower log-odds than Homo (i.e., humans higher).
        # We'll check how many non-human genus params are significantly negative.
        neg_and_sig = [n for n, s in summaries.items() if (s["coef"] < 0 and s["significant_0.05"])]
        neg_not_sig = [n for n, s in summaries.items() if (s["coef"] < 0 and not s["significant_0.05"])]
        pos_and_sig = [n for n, s in summaries.items() if (s["coef"] > 0 and s["significant_0.05"])]
        pos_not_sig = [n for n, s in summaries.items() if (s["coef"] > 0 and not s["significant_0.05"])]

        # Decision rules:
        # - If all non-human genus coefficients are negative and significant -> strong support that humans have higher AMTL
        # - If majority negative & significant -> partial support
        # - If mixed or none significant -> inconclusive / no support
        total = len(genus_params)
        count_neg_sig = len(neg_and_sig)
        if total > 0 and count_neg_sig == total:
            conclusion_code = "yes"
            conclusion_text = ("All non-human genus coefficients are negative and statistically significant. "
                               "This indicates Homo sapiens (the omitted reference) have higher AMTL than each non-human genus.")
        elif total > 0 and count_neg_sig >= math.ceil(total * 0.66):
            conclusion_code = "yes_partial"
            conclusion_text = ("Most non-human genus coefficients are negative and statistically significant, "
                               "suggesting Homo sapiens tend to have higher AMTL than the other genera, though not uniformly.")
        elif count_neg_sig > 0:
            conclusion_code = "inconclusive"
            conclusion_text = ("Some non-human genus coefficients are negative and significant while others are not; "
                               "evidence that Homo sapiens have higher AMTL is mixed/inconclusive.")
        else:
            # if several positive significant ones exist, that would indicate humans lower, otherwise inconclusive
            if len(pos_and_sig) >= 1:
                conclusion_code = "no"
                conclusion_text = ("One or more non-human genus coefficients are positive and significant, "
                                   "indicating those genera have higher AMTL than the reference (Homo), so humans do not have higher AMTL.")
            else:
                conclusion_code = "inconclusive"
                conclusion_text = ("No consistent significant pattern across non-human genus coefficients; cannot conclude that Homo sapiens have higher AMTL.")

        return {
            "object": {
                "method_used": "genus_dummies_vs_reference",
                "parameters_compared": summaries,
                "counts": {
                    "total_genus_params": total,
                    "negative_and_significant": count_neg_sig,
                    "negative_not_significant": len(neg_not_sig),
                    "positive_and_significant": len(pos_and_sig),
                    "positive_not_significant": len(pos_not_sig)
                },
                "conclusion_code": conclusion_code,
                "conclusion_text": conclusion_text
            },
            "description": ("Homo sapiens appears to be the omitted reference level; inspected other genus coefficients "
                            "which represent difference (other_genus - Homo). Negative coefficients indicate lower log-odds "
                            "in the non-human genus relative to Homo (i.e., humans higher).")
        }

    # If we reach here, we couldn't identify genus-related parameters or is_human
    return {
        "object": None,
        "description": "Could not identify an 'is_human' predictor or genus-related parameters in the model output. Please ensure the model included 'genus' (dummies) or an 'is_human' indicator."
    }