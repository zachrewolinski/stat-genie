def extract_final_answer(model_output):
    """
    Extract statistics for the SkinDark coefficient from the model_output produced
    by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coef, SE, z, p, IRR, 95% CI, model info)
      - "description": a short plain-language interpretation of the results in context.

    The function is defensive about which elements are present in model_output:
    it prefers clustered_results (params & bse) if available, otherwise falls back to
    the model's own params and bse.
    """
    import math
    import numpy as np

    # Prepare defaults
    out = {
        "coef": np.nan,
        "se": np.nan,
        "z": np.nan,
        "p_value": np.nan,
        "IRR": np.nan,
        "IRR_CI_95": (np.nan, np.nan),
        "model_family": model_output.get("model_family") if isinstance(model_output, dict) else None,
        "dispersion": model_output.get("dispersion") if isinstance(model_output, dict) else None,
        "note": None
    }

    # Helper to extract series-like params and bse
    params = None
    bse = None
    try:
        # Prefer clustered results if present
        clustered = model_output.get("clustered_results") if isinstance(model_output, dict) else None
        if clustered is not None and hasattr(clustered, "params") and hasattr(clustered, "bse"):
            params = clustered.params
            bse = clustered.bse
        else:
            # Fallback to final_model params / bse
            final_model = model_output.get("final_model") if isinstance(model_output, dict) else None
            if final_model is not None:
                # statsmodels results: params attribute and bse attribute available
                params = getattr(final_model, "params", None)
                bse = getattr(final_model, "bse", None)
    except Exception:
        params = None
        bse = None

    if params is None or bse is None:
        out["note"] = "Could not locate params/bse in model_output."
        return {"object": out, "description": out["note"]}

    # Ensure SkinDark is present
    if "SkinDark" not in params.index:
        out["note"] = "SkinDark parameter not present in model results."
        return {"object": out, "description": out["note"]}

    # Extract coefficient and standard error (prefer numeric extraction)
    try:
        coef = float(params["SkinDark"])
        se = float(bse["SkinDark"])
    except Exception as e:
        out["note"] = f"Failed to coerce coef/SE to float: {e}"
        return {"object": out, "description": out["note"]}

    # z-statistic and two-sided p-value using normal approximation
    z = coef / se if se != 0 else np.nan
    # two-sided p-value via complementary error function: p = erfc(|z|/sqrt(2))
    p_value = math.erfc(abs(z) / math.sqrt(2)) if not math.isnan(z) else np.nan

    # Incidence Rate Ratio (IRR) and 95% CI on IRR scale
    irr = float(math.exp(coef))
    ci_lower = float(math.exp(coef - 1.96 * se))
    ci_upper = float(math.exp(coef + 1.96 * se))

    # Populate output
    out.update({
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "IRR": irr,
        "IRR_CI_95": (ci_lower, ci_upper)
    })

    # Build a concise interpretation
    signif = (p_value < 0.05) if (not math.isnan(p_value)) else False
    if math.isnan(irr):
        desc = "Could not compute IRR for SkinDark."
    else:
        desc = (
            f"Estimated effect of SkinDark (dark vs light): coefficient = {coef:.4f}, "
            f"SE = {se:.4f}, z = {z:.3f}, two-sided p = {p_value:.3f}. "
            f"Incidence rate ratio (IRR) = {irr:.3f} "
            f"(95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
        )
        if signif:
            desc += "This indicates a statistically significant association at α=0.05: dark-skinned players have a higher rate of red cards compared with light-skinned players in this model."
        else:
            desc += "This association is not statistically significant at α=0.05."

    # Return structured object and description
    return {"object": out, "description": desc}