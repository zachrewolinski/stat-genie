def extract_final_answer(model_output):
    """
    Extract interpretable summary statistics from the modeling output.

    Expects model_output to be a dict with keys:
      - 'ols_fish_per_hour' : statsmodels OLS RegressionResultsWrapper (or None)
      - 'poisson_glm_rate'  : statsmodels GLMResultsWrapper (Poisson with offset) (or None)

    Returns a dictionary with keys:
      - "object": a structured dict containing numeric summaries for each model
      - "description": a short human-readable interpretation of main results
    """
    import numpy as np

    summary = {"ols": None, "poisson": None}
    descriptions = []

    # Helper to build coefficient summary dict from a results object
    def coef_table(results):
        # returns dict param -> {coef, se, stat, pvalue, ci_low, ci_high}
        table = {}
        try:
            params = results.params
            bse = results.bse
            pvalues = results.pvalues
            # conf_int returns DataFrame with two columns [0]=low, [1]=high
            ci = results.conf_int()
            # statistic name differs by model: tvalues for OLS, params/others for GLM (z or t)
            stat_name = getattr(results, "tvalues", None)
            if stat_name is None:
                stat_name = getattr(results, "prsquared", None)  # fallback (not ideal)
            for name in params.index:
                coef = float(params[name])
                se = float(bse[name]) if name in bse.index else None
                pval = float(pvalues[name]) if name in pvalues.index else None
                ci_low = float(ci.loc[name, 0]) if name in ci.index else None
                ci_high = float(ci.loc[name, 1]) if name in ci.index else None
                stat = None
                if hasattr(results, "tvalues") and name in results.tvalues.index:
                    stat = float(results.tvalues[name])
                elif hasattr(results, "zvalues") and name in results.zvalues.index:
                    stat = float(results.zvalues[name])
                table[name] = {
                    "coef": coef,
                    "se": se,
                    "stat": stat,
                    "pvalue": pval,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
        except Exception as e:
            # If anything fails, return empty dict
            return {}
        return table

    # Process OLS model (fish_per_hour)
    ols_res = model_output.get("ols_fish_per_hour") if isinstance(model_output, dict) else None
    if ols_res is not None:
        try:
            ols_coefs = coef_table(ols_res)
            # Identify significant predictors (alpha = 0.05), exclude constant
            sig_preds = []
            for name, stats in ols_coefs.items():
                if name.lower() in ("const", "intercept"):
                    continue
                p = stats.get("pvalue")
                if p is not None and p < 0.05:
                    direction = "increase" if stats["coef"] > 0 else "decrease"
                    sig_preds.append({"variable": name, "coef": stats["coef"], "pvalue": p, "direction": direction})
            # Baseline interpretation: OLS intercept is fish_per_hour when predictors == 0
            intercept_name = "const" if "const" in ols_coefs else None
            baseline = None
            if intercept_name:
                baseline = ols_coefs[intercept_name]["coef"]
            ols_summary = {
                "nobs": int(getattr(ols_res, "nobs", -1)) if hasattr(ols_res, "nobs") else None,
                "coefficients": ols_coefs,
                "significant_predictors": sig_preds,
                "baseline_fish_per_hour_when_predictors_zero": baseline,
            }
            summary["ols"] = ols_summary

            # Short description piece
            if sig_preds:
                s = "OLS: Significant predictors of fish_per_hour (alpha=0.05): " + ", ".join(
                    [f"{p['variable']} (coef={p['coef']:.3g}, p={p['pvalue']:.3g}, {p['direction']})" for p in sig_preds]
                )
            else:
                s = "OLS: No predictors were statistically significant at alpha=0.05."
            if baseline is not None:
                s += f" Baseline (all predictors = 0) fish/hour (OLS intercept) = {baseline:.3g}."
            descriptions.append(s)
        except Exception as e:
            summary["ols"] = None
            descriptions.append(f"OLS: failed to extract summary ({e}).")
    else:
        descriptions.append("OLS: model not available in model_output.")

    # Process Poisson GLM model (fish_caught with log(hours) offset -> models rate per hour)
    poisson_res = model_output.get("poisson_glm_rate") if isinstance(model_output, dict) else None
    if poisson_res is not None:
        try:
            pois_coefs = coef_table(poisson_res)
            # For Poisson, interpret as rate ratios: exp(coef)
            for name, stats in pois_coefs.items():
                coef = stats["coef"]
                ci_low = stats["ci_low"]
                ci_high = stats["ci_high"]
                try:
                    rr = float(np.exp(coef)) if coef is not None else None
                    rr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                    rr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
                except Exception:
                    rr = rr_ci_low = rr_ci_high = None
                stats["rate_ratio"] = rr
                stats["rate_ratio_ci_low"] = rr_ci_low
                stats["rate_ratio_ci_high"] = rr_ci_high

            # Identify significant predictors (alpha = 0.05), exclude constant
            sig_preds = []
            for name, stats in pois_coefs.items():
                if name.lower() in ("const", "intercept"):
                    continue
                p = stats.get("pvalue")
                if p is not None and p < 0.05:
                    direction = "increase" if stats["rate_ratio"] > 1 else "decrease"
                    sig_preds.append({
                        "variable": name,
                        "coef": stats["coef"],
                        "rate_ratio": stats["rate_ratio"],
                        "rate_ratio_ci": (stats["rate_ratio_ci_low"], stats["rate_ratio_ci_high"]),
                        "pvalue": p,
                        "direction": direction
                    })
            # Baseline rate per hour: exp(intercept) (because offset = log(hours))
            intercept_name = "const" if "const" in pois_coefs else None
            baseline_rate = None
            if intercept_name:
                intercept_coef = pois_coefs[intercept_name]["coef"]
                if intercept_coef is not None:
                    baseline_rate = float(np.exp(intercept_coef))
            poisson_summary = {
                "nobs": int(getattr(poisson_res, "nobs", -1)) if hasattr(poisson_res, "nobs") else None,
                "coefficients": pois_coefs,
                "significant_predictors": sig_preds,
                "baseline_rate_per_hour_when_predictors_zero": baseline_rate,
                "note": "Poisson coefficients are on log(rate) scale; reported 'rate_ratio' = exp(coef)."
            }
            summary["poisson"] = poisson_summary

            # Short description piece
            if sig_preds:
                s = "Poisson: Significant predictors of fish-catch rate (alpha=0.05): " + "; ".join(
                    [f"{p['variable']}: rate_ratio={p['rate_ratio']:.3g} (p={p['pvalue']:.3g}, {p['direction']})" for p in sig_preds]
                )
            else:
                s = "Poisson: No predictors were statistically significant at alpha=0.05."
            if baseline_rate is not None:
                s += f" Baseline predicted fish/hour (all predictors = 0) = {baseline_rate:.3g}."
            s += " Interpret Poisson rate_ratio as multiplicative change in fish caught per hour."
            descriptions.append(s)
        except Exception as e:
            summary["poisson"] = None
            descriptions.append(f"Poisson: failed to extract summary ({e}).")
    else:
        descriptions.append("Poisson: model not available in model_output.")

    # Combine descriptions into one short explanation
    full_description = " | ".join(descriptions)

    return {"object": summary, "description": full_description}