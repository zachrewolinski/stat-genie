def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, t-stats, p-values, 95% CIs, and an interpretable effect
    (approx percent change in deaths+1) for the key predictors:
      - masfem_z (continuous standardized femininity; higher = more feminine)
      - female_name (binary indicator; 1 = female name)
    
    Returns:
      {
        "object": {
          "model_masfem": { ...stats... , "supports_hypothesis": True/False/None },
          "model_female_name": { ...stats... , "supports_hypothesis": True/False/None }
        },
        "description": "Brief interpretation of the extracted stats and how they relate to the hypothesis."
      }
    """
    import numpy as np

    out = {"model_masfem": None, "model_female_name": None}
    msg_parts = []

    # Helper to extract stats for a given fitted model and variable name
    def extract_from_model(result, varname):
        if result is None:
            return None
        res = {}
        try:
            params = result.params
            if varname not in params.index:
                return None
            beta = float(params[varname])
            se = float(result.bse[varname]) if varname in result.bse.index else None
            tval = float(result.tvalues[varname]) if varname in result.tvalues.index else None
            pval = float(result.pvalues[varname]) if varname in result.pvalues.index else None
            ci = result.conf_int().loc[varname].tolist() if varname in result.conf_int().index else [None, None]
            nobs = int(getattr(result, "nobs", result.df_resid + result.df_model + 1))  # fallback
            # Interpret effect on log(deaths + 1): convert to percent change in deaths+1
            try:
                pct_change = (np.exp(beta) - 1) * 100.0
            except Exception:
                pct_change = None
            # Determine whether effect supports the hypothesis:
            # Hypothesis: more feminine names -> fewer precautions -> more fatalities.
            # So we expect a positive coefficient (higher femininity -> higher log_deaths).
            if pval is None:
                supports = None
            else:
                supports = (beta > 0) and (pval < 0.05)
            res.update({
                "variable": varname,
                "coefficient": beta,
                "std_error": se,
                "t_value": tval,
                "p_value": pval,
                "ci_95": ci,
                "n_obs": nobs,
                "percent_change_deaths_plus1": pct_change,
                "supports_hypothesis": supports
            })
            return res
        except Exception:
            return None

    model_masfem = model_output.get("model_masfem") or model_output.get("model_masfem".lower())
    model_female = model_output.get("model_female_name") or model_output.get("model_female_name".lower())

    out["model_masfem"] = extract_from_model(model_masfem, "masfem_z")
    out["model_female_name"] = extract_from_model(model_female, "female_name")

    # Build a short human-readable description
    def make_desc(entry, label):
        if entry is None:
            return f"{label}: variable not present in the fitted model or model missing."
        coef = entry["coefficient"]
        p = entry["p_value"]
        pct = entry["percent_change_deaths_plus1"]
        supports = entry["supports_hypothesis"]
        ci = entry["ci_95"]
        s = (f"{label}: coefficient={coef:.4g}, 95% CI=[{ci[0]:.4g}, {ci[1]:.4g}], p={p:.4g}, "
             f"approx % change in (deaths+1) per unit = {pct:.2f}% (exp(coef)-1). ")
        if supports is True:
            s += "This result is statistically significant (p<0.05) and in the direction that supports the hypothesis: more feminine names -> higher fatalities (consistent with fewer precautions)."
        elif supports is False:
            # If p >= .05 or coefficient in opposite direction
            if coef > 0:
                s += "Coefficient is positive but not statistically significant at the 0.05 level, so evidence is weak/insufficient to support the hypothesis."
            else:
                s += "Coefficient is not in the hypothesized direction (or not statistically significant), so it does not support the hypothesis."
        else:
            s += "Insufficient information to determine statistical support."
        return s

    if out["model_masfem"] is not None:
        msg_parts.append(make_desc(out["model_masfem"], "Model with masfem_z"))
    else:
        msg_parts.append("Model with masfem_z: not available or variable missing.")

    if out["model_female_name"] is not None:
        msg_parts.append(make_desc(out["model_female_name"], "Model with female_name"))
    else:
        msg_parts.append("Model with female_name: not available or variable missing.")

    description = " ".join(msg_parts)

    return {"object": out, "description": description}