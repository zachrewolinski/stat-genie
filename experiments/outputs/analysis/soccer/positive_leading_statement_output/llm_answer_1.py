def extract_final_answer(model_output):
    """
    Extract key statistics about the IsDark effect from model_output.
    Returns a dictionary with:
      - "object": dict with extracted stats for Negative Binomial (rate/IRR)
                  and Logistic (odds/OR) models when available.
      - "description": short plain-language interpretation of each extracted stat.
    """
    import numpy as np

    result = {"object": {}, "description": ""}

    # Helper to safely get an estimator (prefer cluster-robust if present)
    def _get_model(key_cluster, key_plain):
        if key_cluster in model_output and model_output[key_cluster] is not None:
            return model_output[key_cluster]
        if key_plain in model_output and model_output[key_plain] is not None:
            return model_output[key_plain]
        return None

    # 1) Negative Binomial results (effect on redCards; exposure=games)
    nb_res = _get_model("nb_model_cluster", "nb_model")
    if nb_res is None:
        result["object"]["negative_binomial"] = None
        nb_text = "Negative binomial model not found in model_output."
    else:
        try:
            params = nb_res.params
            bse = nb_res.bse
            pvals = nb_res.pvalues
            ci = nb_res.conf_int()  # default 95% CI on coefficient (log) scale

            if "IsDark" not in params.index:
                raise KeyError("IsDark not in negative binomial model parameters.")

            coef = float(params.loc["IsDark"])
            se = float(bse.loc["IsDark"]) if "IsDark" in bse.index else None
            p = float(pvals.loc["IsDark"]) if "IsDark" in pvals.index else None
            ci_low_log, ci_upp_log = float(ci.loc["IsDark", 0]), float(ci.loc["IsDark", 1])

            irr = float(np.exp(coef))
            irr_ci_low = float(np.exp(ci_low_log))
            irr_ci_upp = float(np.exp(ci_upp_log))
            pct_change = (irr - 1.0) * 100.0  # percent change in rate

            sig = (p is not None) and (p < 0.05)

            result["object"]["negative_binomial"] = {
                "coef_log_rate": coef,
                "se": se,
                "p_value": p,
                "ci_log_rate": [ci_low_log, ci_upp_log],
                "IRR": irr,
                "IRR_CI": [irr_ci_low, irr_ci_upp],
                "percent_change_rate": pct_change,
                "significant_p_lt_0_05": bool(sig),
            }

            nb_text = (
                "Negative binomial model: coefficient is on the log-rate scale. "
                f"Estimated log-rate coef for IsDark = {coef:.4f} (SE = {se:.4f}, p = {p:.4g}). "
                f"Equivalently, IRR = {irr:.3f} (95% CI [{irr_ci_low:.3f}, {irr_ci_upp:.3f}]), "
                f"which corresponds to a {pct_change:.1f}% change in the expected red-card rate per game for dark vs light players. "
                f"{'Statistically significant (p < 0.05).' if sig else 'Not statistically significant (p >= 0.05).'}"
            )
        except Exception as e:
            result["object"]["negative_binomial"] = None
            nb_text = f"Failed to extract negative binomial stats: {e}"

    # 2) Logistic results (sensitivity: any_red)
    logit_res = _get_model("logit_model_cluster", "logit_model")
    if logit_res is None:
        result["object"]["logistic_any_red"] = None
        logit_text = "Logistic model not found in model_output."
    else:
        try:
            params = logit_res.params
            bse = logit_res.bse
            pvals = logit_res.pvalues
            ci = logit_res.conf_int()

            if "IsDark" not in params.index:
                raise KeyError("IsDark not in logistic model parameters.")

            coef = float(params.loc["IsDark"])
            se = float(bse.loc["IsDark"]) if "IsDark" in bse.index else None
            p = float(pvals.loc["IsDark"]) if "IsDark" in pvals.index else None
            ci_low_log, ci_upp_log = float(ci.loc["IsDark", 0]), float(ci.loc["IsDark", 1])

            or_val = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low_log))
            or_ci_upp = float(np.exp(ci_upp_log))
            pct_change_or = (or_val - 1.0) * 100.0

            sig = (p is not None) and (p < 0.05)

            result["object"]["logistic_any_red"] = {
                "coef_log_odds": coef,
                "se": se,
                "p_value": p,
                "ci_log_odds": [ci_low_log, ci_upp_log],
                "OR": or_val,
                "OR_CI": [or_ci_low, or_ci_upp],
                "percent_change_odds": pct_change_or,
                "significant_p_lt_0_05": bool(sig),
            }

            logit_text = (
                "Logistic model (any_red): coefficient is on the log-odds scale. "
                f"Estimated log-odds coef for IsDark = {coef:.4f} (SE = {se:.4f}, p = {p:.4g}). "
                f"Equivalently, OR = {or_val:.3f} (95% CI [{or_ci_low:.3f}, {or_ci_upp:.3f}]), "
                f"i.e., a {pct_change_or:.1f}% change in odds of receiving any red card for dark vs light players. "
                f"{'Statistically significant (p < 0.05).' if sig else 'Not statistically significant (p >= 0.05).'}"
            )
        except Exception as e:
            result["object"]["logistic_any_red"] = None
            logit_text = f"Failed to extract logistic stats: {e}"

    # Compose final description
    descriptions = []
    descriptions.append(nb_text)
    descriptions.append(logit_text)
    result["description"] = " ".join(descriptions)

    return result