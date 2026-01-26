def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and approximate percent-change
    interpretations for the two key predictors (masfem_z and gender_female) from the
    supplied model_output dict.

    Returns:
      {
        "object": {
          "n_obs": int,
          "models": {
            "ols": { "masfem_z": {...}, "gender_female": {...} } or None,
            "negative_binomial": { ... } or None,
            "poisson": { ... } or None
          }
        },
        "description": str  # brief interpretation and final conclusion about the hypothesis
      }
    """
    import numpy as np

    result = {
        "n_obs": model_output.get("n_obs"),
        "models": {}
    }

    # Which models to extract
    model_keys = ["ols", "negative_binomial", "poisson"]
    predictors = ["masfem_z", "gender_female"]

    for mk in model_keys:
        mod = model_output.get(mk)
        if mod is None:
            result["models"][mk] = None
            continue

        try:
            params = mod.params
            pvalues = mod.pvalues
            conf = mod.conf_int()
        except Exception as e:
            # If any attribute missing, record None
            result["models"][mk] = None
            continue

        model_info = {}
        for pred in predictors:
            if pred in params.index:
                beta = float(params[pred])
                p = float(pvalues[pred])
                # conf may be DataFrame indexed by param names
                try:
                    ci_low = float(conf.loc[pred].iloc[0])
                    ci_high = float(conf.loc[pred].iloc[1])
                except Exception:
                    # fallback if indexing differs
                    ci_low = float(conf.iloc[params.index.get_loc(pred), 0])
                    ci_high = float(conf.iloc[params.index.get_loc(pred), 1])

                # For interpretation on log(1+deaths) (OLS) or log link GLMs,
                # approximate percent change in (1 + alldeaths) per one-unit increase:
                pct_change = (np.exp(beta) - 1.0) * 100.0

                model_info[pred] = {
                    "coef": beta,
                    "pvalue": p,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "approx_percent_change_in_1_plus_deaths": pct_change,
                    "significant_at_0.05": bool(p < 0.05)
                }
        result["models"][mk] = model_info

    # Form a concise interpretation focusing on the primary model (OLS)
    desc_lines = []
    desc_lines.append(f"Number of observations used: {result['n_obs']}")
    ols_info = result["models"].get("ols")
    if ols_info is None:
        desc_lines.append("OLS results not available.")
    else:
        if "masfem_z" in ols_info:
            m = ols_info["masfem_z"]
            desc_lines.append(
                "Primary (OLS on log(1+deaths)) — masfem_z (standardized femininity): "
                f"coef = {m['coef']:.4f}, 95% CI = [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}], "
                f"p = {m['pvalue']:.3f}. "
                f"This corresponds to an approximate {m['approx_percent_change_in_1_plus_deaths']:.1f}% "
                "change in (1 + fatalities) per 1 SD increase in perceived femininity."
            )
            if m["significant_at_0.05"]:
                if m["coef"] < 0:
                    desc_lines.append("Interpretation: Statistically significant negative effect — higher femininity is associated with fewer fatalities (supports the hypothesis).")
                else:
                    desc_lines.append("Interpretation: Statistically significant positive effect — higher femininity is associated with more fatalities (contradicts the hypothesis).")
            else:
                if m["coef"] < 0:
                    desc_lines.append("Interpretation: Point estimate is negative (consistent with the hypothesis) but not statistically significant at p<0.05.")
                else:
                    desc_lines.append("Interpretation: Point estimate is positive (not consistent with the hypothesis) and not statistically significant at p<0.05.")
        else:
            desc_lines.append("masfem_z not found in OLS results.")

    # Brief note about the binary gender predictor
    if ols_info and "gender_female" in ols_info:
        g = ols_info["gender_female"]
        desc_lines.append(
            "OLS — gender_female (1=female name): "
            f"coef = {g['coef']:.4f}, 95% CI = [{g['ci_lower']:.4f}, {g['ci_upper']:.4f}], p = {g['pvalue']:.3f}. "
            f"Approx percent change in (1 + fatalities) = {g['approx_percent_change_in_1_plus_deaths']:.1f}%."
        )
        if g["significant_at_0.05"]:
            desc_lines.append("Interpretation: gender_female effect is statistically significant in OLS.")
        else:
            desc_lines.append("Interpretation: gender_female effect is not statistically significant in OLS.")

    # Short robustness summary: check consistency of sign and significance across GLMs
    rob_lines = []
    for mk in ["negative_binomial", "poisson"]:
        info = result["models"].get(mk)
        if info is None:
            rob_lines.append(f"{mk}: model not available or failed.")
            continue
        if "masfem_z" in info:
            mm = info["masfem_z"]
            s = "sig" if mm["significant_at_0.05"] else "n.s."
            rob_lines.append(f"{mk}: masfem_z coef={mm['coef']:.4f} (approx {mm['approx_percent_change_in_1_plus_deaths']:.1f}%), p={mm['pvalue']:.3f} [{s}]")
    if rob_lines:
        desc_lines.append("Robustness checks: " + " ; ".join(rob_lines))

    # Final plain-language conclusion
    # Use OLS as primary: if OLS masfem_z is negative and significant -> yes, else no strong evidence
    final = "Final conclusion: "
    if ols_info and "masfem_z" in ols_info:
        m = ols_info["masfem_z"]
        if m["coef"] < 0 and m["significant_at_0.05"]:
            final += "Yes — there is statistically significant evidence that more feminine hurricane names are associated with fewer fatalities (consistent with the hypothesis)."
        elif m["coef"] < 0 and not m["significant_at_0.05"]:
            final += "No strong evidence — point estimate is in the hypothesized direction (feminine -> fewer fatalities) but is not statistically significant in the primary model."
        elif m["coef"] >= 0 and m["significant_at_0.05"]:
            final += "No — the primary model shows a statistically significant association in the opposite direction (feminine names associated with higher fatalities)."
        else:
            final += "No — the primary model does not show a statistically significant effect."
    else:
        final += "Insufficient information (OLS info missing)."

    desc_lines.append(final)

    return {"object": result, "description": " ".join(desc_lines)}