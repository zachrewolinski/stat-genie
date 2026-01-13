def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and interpretable rate ratios
    from the fitted GLM (Poisson or Negative Binomial) stored in model_output.
    
    Returns a dict with:
      - "object": a dict of numeric results (coefficients, p-values, conf. intervals,
                  exponentiated coefficients = rate ratios, mean observed rate, dispersion, family)
      - "description": a human-readable summary explaining the key results and how to interpret them,
                       including a short interpretation for LiveBait and HasCamper when present.
    """
    import numpy as np
    import pandas as pd

    # Find the fitted results object
    res = model_output.get('final_result') or model_output.get('negative_binomial_result') or model_output.get('poisson_result')
    if res is None:
        return {
            "object": None,
            "description": "No fitted model object found in model_output. Expected key 'final_result' or similar."
        }

    # Extract basic statistics
    try:
        params = res.params          # pandas Series
        pvalues = res.pvalues        # pandas Series
        conf = res.conf_int()        # DataFrame with two columns [lower, upper]
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract params/pvalues/conf_int from model result: {e}"
        }

    # Exponentiate to get rate ratios (multiplicative effect on fish-per-hour)
    rate_ratios = np.exp(params)
    rr_conf = np.exp(conf)

    # Helper to convert pandas objects to plain Python floats/lists for JSON-compatibility
    def series_to_dict(s):
        return {str(k): float(v) for k, v in s.items()}

    def confdf_to_dict(df):
        return {str(idx): [float(df.loc[idx].iloc[0]), float(df.loc[idx].iloc[1])] for idx in df.index}

    coeffs_dict = series_to_dict(params)
    pvals_dict = series_to_dict(pvalues)
    conf_dict = confdf_to_dict(conf)
    rr_dict = series_to_dict(rate_ratios)
    rr_conf_dict = confdf_to_dict(rr_conf)

    # Interpretation for key binary predictors if present
    interpretation = {}
    for var in ['LiveBait', 'HasCamper']:
        if var in coeffs_dict:
            rr = rr_dict[var]
            ci = rr_conf_dict[var]
            p = pvals_dict[var]
            signif = p < 0.05
            interpretation[var] = {
                "rate_ratio": rr,
                "95%_CI": ci,
                "p_value": p,
                "significant_at_0.05": bool(signif),
                "interpretation": (
                    f"A rate_ratio > 1 means higher fish-per-hour when {var}=1 (vs 0); "
                    f"here rate_ratio={rr:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f}), p={p:.3g}. "
                    + ("Statistically significant." if signif else "Not statistically significant.")
                )
            }

    # Additional summaries from model_output if available
    mean_rate = model_output.get('mean_rate_per_hour')
    dispersion = model_output.get('dispersion')
    used_family = model_output.get('used_family')

    # Assemble final object
    result_object = {
        "coefficients": coeffs_dict,
        "p_values": pvals_dict,
        "conf_int": conf_dict,
        "rate_ratios": rr_dict,
        "rate_ratio_conf_int": rr_conf_dict,
        "interpretation_key_vars": interpretation,
        "mean_rate_per_hour_observed": float(mean_rate) if mean_rate is not None else None,
        "dispersion": float(dispersion) if dispersion is not None else None,
        "used_family": used_family
    }

    # Build human-readable description
    desc_lines = []
    desc_lines.append(f"Model family used: {used_family}. Dispersion (Poisson check): {dispersion:.3f}" if dispersion is not None else f"Model family used: {used_family}.")
    if mean_rate is not None:
        desc_lines.append(f"Observed mean fish-per-hour (overall): {mean_rate:.3f}.")
    desc_lines.append("Returned quantities:")
    desc_lines.append(" - 'coefficients': log-rate coefficients (log fish-per-hour when offset applied).")
    desc_lines.append(" - 'rate_ratios': exponentiated coefficients = multiplicative change in fish-per-hour per unit increase in predictor.")
    desc_lines.append(" - 'conf_int' and 'rate_ratio_conf_int': 95% confidence intervals on coefficients and rate ratios.")
    desc_lines.append(" - 'p_values': p-values for each coefficient.")
    desc_lines.append("")
    if interpretation:
        desc_lines.append("Quick interpretation for binary predictors included in the model:")
        for var, info in interpretation.items():
            desc_lines.append(f" * {var}: {info['interpretation']}")
    else:
        desc_lines.append("No LiveBait/HasCamper variables found in coefficients to summarize.")

    description = " ".join(desc_lines)

    return {
        "object": result_object,
        "description": description
    }