def extract_final_answer(model_output):
    """
    Extracts and summarizes key statistics from the provided model_output dictionary
    (expects keys 'ols' and 'poisson' with statsmodels results wrappers).
    
    Returns a dict with:
      - "object": a structured dictionary of coefficients, SEs, p-values, confidence intervals,
                  exponentiated effects (rate ratios), and percent-change interpretations
                  for each predictor from both models; plus baseline rate estimates.
      - "description": a short text explaining what the numbers mean in the context.
    """
    import numpy as np

    # Helper to extract from a statsmodels results object
    def summarize_results(res, model_type="poisson"):
        summary = {}
        try:
            params = res.params
            bse = res.bse
            pvalues = res.pvalues
            ci = res.conf_int()
        except Exception as e:
            raise RuntimeError(f"Unable to extract standard summaries from model result: {e}")

        for name in params.index:
            coef = float(params[name])
            se = float(bse.get(name, np.nan))
            p = float(pvalues.get(name, np.nan))
            # conf_int returns DataFrame-like with two columns [lower, upper]
            try:
                lower, upper = float(ci.loc[name, 0]), float(ci.loc[name, 1])
            except Exception:
                # fallback if ci is structured differently
                lower, upper = (np.nan, np.nan)

            # Interpretations:
            # For both Poisson (log link with offset) and OLS on log(rate), coef is on log-rate scale.
            # exponentiated coef = multiplicative effect on rate (rate ratio).
            exp_coef = np.exp(coef)
            exp_lower = np.exp(lower) if not np.isnan(lower) else np.nan
            exp_upper = np.exp(upper) if not np.isnan(upper) else np.nan
            pct_change = (exp_coef - 1.0) * 100.0
            pct_lower = (exp_lower - 1.0) * 100.0 if not np.isnan(exp_lower) else np.nan
            pct_upper = (exp_upper - 1.0) * 100.0 if not np.isnan(exp_upper) else np.nan

            summary[name] = {
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "conf_int": [lower, upper],
                "exp(coef)_rate_ratio": exp_coef,
                "exp(conf_int)_rate_ratio": [exp_lower, exp_upper],
                "pct_change_in_rate": pct_change,
                "pct_change_conf_int": [pct_lower, pct_upper]
            }
        return summary

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing at least 'ols' and 'poisson' results.")

    out = {"object": {}, "description": ""}

    # Summarize models if present
    models_present = []
    if "poisson" in model_output and model_output["poisson"] is not None:
        try:
            poisson_summary = summarize_results(model_output["poisson"], model_type="poisson")
            out["object"]["poisson"] = poisson_summary
            models_present.append("poisson")
            # baseline poisson rate per hour when predictors = 0: exp(intercept)
            if "const" in poisson_summary:
                intercept = poisson_summary["const"]["coef"]
                baseline_rate_poisson = float(np.exp(intercept))
            else:
                baseline_rate_poisson = None
            out["object"]["baseline_rate_poisson_per_hour"] = baseline_rate_poisson
        except Exception as e:
            out["object"]["poisson_error"] = str(e)

    if "ols" in model_output and model_output["ols"] is not None:
        try:
            ols_summary = summarize_results(model_output["ols"], model_type="ols")
            out["object"]["ols"] = ols_summary
            models_present.append("ols")
            # baseline OLS predicted rate per hour:
            # Note: OLS was fit on log((fish_caught + 0.1)/hours). So intercept back-transformed gives an estimate
            # of typical fish_per_hour at reference values, but slight bias due to the +0.1 offset.
            if "const" in ols_summary:
                intercept = ols_summary["const"]["coef"]
                baseline_rate_ols = float(np.exp(intercept))
            else:
                baseline_rate_ols = None
            out["object"]["baseline_rate_ols_backtransformed_per_hour"] = baseline_rate_ols
        except Exception as e:
            out["object"]["ols_error"] = str(e)

    if len(models_present) == 0:
        raise ValueError("No usable models found in model_output. Expecting keys 'ols' and/or 'poisson'.")

    # Provide a concise description of the interpretation
    desc_lines = []
    desc_lines.append("This output summarizes model estimates for factors associated with fish caught per hour.")
    desc_lines.append("- For both models, coefficients are on the log-rate scale. Exponentiating a coefficient")
    desc_lines.append("  yields a rate ratio: exp(coef) is the multiplicative change in fish-per-hour for a one-unit")
    desc_lines.append("  increase in the predictor (holding others constant).")
    desc_lines.append("- 'pct_change_in_rate' reports (exp(coef)-1)*100, the percent change in catch-rate per hour.")
    desc_lines.append("- Confidence intervals are provided both on the log scale and after exponentiation (rate ratios).")
    desc_lines.append("- The baseline_rate_* values are predicted fish-per-hour when all predictors equal zero (OLS baseline")
    desc_lines.append("  uses the logged outcome that included a +0.1 offset, so treat it as an approximate reference).")
    desc_lines.append("Use the p-values and confidence intervals to judge statistical evidence of effects (p < 0.05 often used).")

    out["description"] = " ".join(desc_lines)

    return out