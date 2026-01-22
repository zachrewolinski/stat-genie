def extract_final_answer(model_output):
    """
    Extract the effect of 'children_binary' on extramarital affairs from the models
    returned in `model_output`. Prefer the logistic model if it converged; otherwise
    try to use the ZINB model if it converged. If a model did not converge, its
    estimates are treated as unreliable and noted in the description.

    Returns a dict with keys:
      - "object": a dict containing numeric results (coef, se, pvalue, CI, odds ratio where applicable)
      - "description": a short plain-language interpretation focused on whether having children
                       is associated with less engagement in extramarital affairs.
    """
    import numpy as np

    result = {"object": None, "description": ""}

    # Helper to format output dictionary
    def make_obj(source, coef, se, pval, ci_low, ci_high, extra=None):
        obj = {
            "model": source,
            "variable": "children_binary",
            "coef": float(coef),
            "se": float(se) if se is not None else None,
            "pvalue": float(pval) if pval is not None else None,
            "ci_lower": float(ci_low) if ci_low is not None else None,
            "ci_upper": float(ci_high) if ci_high is not None else None
        }
        # for logistic, add odds ratio info
        if extra and extra.get("type") == "logit":
            or_val = float(np.exp(coef))
            or_low = float(np.exp(ci_low))
            or_high = float(np.exp(ci_high))
            obj.update({
                "odds_ratio": or_val,
                "odds_ratio_ci_lower": or_low,
                "odds_ratio_ci_upper": or_high
            })
        if extra:
            obj.update(extra)
        return obj

    # 1) Try logistic model first (robust / interpretable)
    logit = model_output.get("logit_model")
    try:
        if logit is not None and getattr(logit, "converged", True):
            # Extract coefficient, se, p-value, conf int for children_binary
            coef = logit.params.loc["children_binary"]
            se = logit.bse.loc["children_binary"]
            pval = logit.pvalues.loc["children_binary"]
            ci = logit.conf_int().loc["children_binary"]
            ci_low, ci_high = ci[0], ci[1]

            result["object"] = make_obj(
                source="logit",
                coef=coef, se=se, pval=pval,
                ci_low=ci_low, ci_high=ci_high,
                extra={"type": "logit"}
            )

            # Interpretation
            # Positive coef => higher odds with children; check significance
            if pval < 0.05:
                sig_text = "statistically significant (p < 0.05)."
            else:
                sig_text = "not statistically significant (p >= 0.05)."

            result["description"] = (
                f"Logistic regression (any affair vs none): children_binary coef = {coef:.4f}, "
                f"SE = {se:.4f}, p = {pval:.3f}. Odds ratio = {np.exp(coef):.3f} "
                f"(95% CI {np.exp(ci_low):.3f} to {np.exp(ci_high):.3f}). This indicates a "
                f"{'positive' if coef>0 else 'negative'} association between having children and "
                f"reporting any extramarital affair, but it is {sig_text}"
            )
            # Also note ZINB convergence status if present
            zinb = model_output.get("zinb_model")
            if zinb is not None and not getattr(zinb, "converged", True):
                result["description"] += " The complementary ZINB model did not converge, so its estimates are unreliable."
            return result
    except Exception:
        # fall through to try ZINB or return failure
        pass

    # 2) If logistic unavailable/unconverged, try ZINB (count model)
    zinb = model_output.get("zinb_model")
    try:
        if zinb is not None and getattr(zinb, "converged", False):
            # For ZINB, children_binary appears both in the inflation and the count parts.
            # We'll extract the count-part coefficient named 'children_binary' if present.
            params = zinb.params
            bse = zinb.bse if hasattr(zinb, "bse") else None
            if "children_binary" in params.index:
                coef = params.loc["children_binary"]
                se = float(bse.loc["children_binary"]) if bse is not None else None
                # p-values/conf_int may not be available; try to get them if possible
                pval = float(zinb.pvalues.loc["children_binary"]) if hasattr(zinb, "pvalues") else None
                ci = zinb.conf_int().loc["children_binary"] if hasattr(zinb, "conf_int") or hasattr(zinb, "conf_int") else (None, None)
                ci_low, ci_high = (float(ci[0]), float(ci[1])) if ci is not None else (None, None)

                result["object"] = make_obj(
                    source="zinb_count",
                    coef=coef, se=se, pval=pval,
                    ci_low=ci_low, ci_high=ci_high
                )

                if pval is not None and pval < 0.05:
                    sig_text = "statistically significant (p < 0.05)."
                else:
                    sig_text = "not statistically significant (or unavailable)."

                result["description"] = (
                    f"ZINB count part: children_binary coef = {coef:.4f}"
                    + (f", SE = {se:.4f}" if se is not None else "")
                    + (f", p = {pval:.3f}" if pval is not None else "")
                    + f". This is {sig_text}"
                )
                return result
            else:
                result["description"] = "ZINB model present and converged but 'children_binary' not found in parameters."
                return result
        else:
            # ZINB not present or did not converge
            # If a summary text mentions 'children_binary', we could try to parse, but safer to report inability.
            zinb_summary = model_output.get("zinb_summary")
            if zinb_summary and "converged: False" in zinb_summary:
                result["description"] = "ZINB model did not converge; estimates from ZINB are unreliable. Logistic model could not be used/extracted."
            else:
                result["description"] = "Neither a converged logistic model nor a converged ZINB model with usable 'children_binary' parameter was found."
            return result
    except Exception:
        result["description"] = "Error extracting parameters from available models."
        return result

    # If reached here, nothing usable was found
    result["description"] = "Could not extract a reliable estimate for 'children_binary' from the provided model output."
    return result