def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of name femininity (z_masfem) on fatalities
    from the provided model_output dict (expected to contain statsmodels result objects).
    
    Returns a dictionary with:
      - "object": a dict containing extracted statistics for each available model:
          * For Negative Binomial models: raw coefficient, se, p-value, 95% CI,
            exponentiated coefficient (IRR) and IRR 95% CI.
          * For OLS (log fatalities): raw coefficient, se, p-value, 95% CI,
            and implied percent change (exp(coef)-1) with CI.
          * Raw means by gender if provided.
      - "description": a plain-language interpretation of the key results (direction,
        statistical significance) for each model present in model_output.
    """
    import numpy as np

    out = {}
    descriptions = []

    def extract_from_model(result, varname):
        """
        Extract coef, se, p, ci for varname from a statsmodels results object.
        Returns dict or raises ValueError if varname not present.
        """
        if not hasattr(result, "params"):
            raise ValueError("Provided result object has no .params attribute")
        params = result.params
        if varname not in params.index:
            raise ValueError(f"Variable '{varname}' not found in model params")
        coef = float(params[varname])
        # standard error and p-value
        bse = float(result.bse[varname]) if hasattr(result, "bse") else None
        pval = float(result.pvalues[varname]) if hasattr(result, "pvalues") else None
        # confidence interval
        try:
            ci_df = result.conf_int()
            # conf_int may be DataFrame with index
            lower, upper = float(ci_df.loc[varname][0]), float(ci_df.loc[varname][1])
        except Exception:
            # fallback: try indexing by position
            try:
                ci_arr = result.conf_int()
                # find position
                idx = list(result.params.index).index(varname)
                lower, upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                lower, upper = None, None

        return {"coef": coef, "se": bse, "pvalue": pval, "ci_lower": lower, "ci_upper": upper}

    # 1) Primary Negative Binomial model: z_masfem
    if "nb_primary" in model_output and not isinstance(model_output["nb_primary"], str):
        try:
            nb = model_output["nb_primary"]
            stats = extract_from_model(nb, "z_masfem")
            # For count models, exponentiate coef -> incidence rate ratio (IRR)
            irr = np.exp(stats["coef"])
            irr_ci_lower = np.exp(stats["ci_lower"]) if stats["ci_lower"] is not None else None
            irr_ci_upper = np.exp(stats["ci_upper"]) if stats["ci_upper"] is not None else None
            out["nb_primary"] = {
                "type": "NegativeBinomial (GLM)",
                "variable": "z_masfem",
                **stats,
                "IRR": float(irr),
                "IRR_ci_lower": float(irr_ci_lower) if irr_ci_lower is not None else None,
                "IRR_ci_upper": float(irr_ci_upper) if irr_ci_upper is not None else None,
            }
            # Interpret
            p = out["nb_primary"]["pvalue"]
            coef = out["nb_primary"]["coef"]
            irr = out["nb_primary"]["IRR"]
            if p is None:
                sig_text = "p-value not available"
            elif p < 0.01:
                sig_text = "highly statistically significant (p < 0.01)"
            elif p < 0.05:
                sig_text = "statistically significant (p < 0.05)"
            elif p < 0.1:
                sig_text = "marginally significant (p < 0.1)"
            else:
                sig_text = "not statistically significant (p >= 0.1)"
            direction = "associated with more fatalities" if coef > 0 else "associated with fewer fatalities" if coef < 0 else "no association"
            descriptions.append(f"NB primary: z_masfem coef={coef:.4f}, IRR={irr:.4f}; {sig_text}; direction: {direction}.")
        except Exception as e:
            out["nb_primary_error"] = str(e)
            descriptions.append(f"NB primary: failed to extract stats ({e}).")

    else:
        descriptions.append("NB primary model not present or is an error message in model_output.")

    # 2) OLS on log(alldeaths)
    if "ols_log_fatalities" in model_output and not isinstance(model_output["ols_log_fatalities"], str):
        try:
            ols = model_output["ols_log_fatalities"]
            stats = extract_from_model(ols, "z_masfem")
            # For log outcome, convert coef to percent change: (exp(coef)-1)*100
            pct_change = (np.exp(stats["coef"]) - 1.0) * 100.0
            pct_ci_lower = (np.exp(stats["ci_lower"]) - 1.0) * 100.0 if stats["ci_lower"] is not None else None
            pct_ci_upper = (np.exp(stats["ci_upper"]) - 1.0) * 100.0 if stats["ci_upper"] is not None else None
            out["ols_log_fatalities"] = {
                "type": "OLS on log(alldeaths)",
                "variable": "z_masfem",
                **stats,
                "pct_change": float(pct_change),
                "pct_change_ci_lower": float(pct_ci_lower) if pct_ci_lower is not None else None,
                "pct_change_ci_upper": float(pct_ci_upper) if pct_ci_upper is not None else None,
            }
            p = out["ols_log_fatalities"]["pvalue"]
            coef = out["ols_log_fatalities"]["coef"]
            if p is None:
                sig_text = "p-value not available"
            elif p < 0.01:
                sig_text = "highly statistically significant (p < 0.01)"
            elif p < 0.05:
                sig_text = "statistically significant (p < 0.05)"
            elif p < 0.1:
                sig_text = "marginally significant (p < 0.1)"
            else:
                sig_text = "not statistically significant (p >= 0.1)"
            direction = "higher fatalities" if coef > 0 else "lower fatalities" if coef < 0 else "no association"
            descriptions.append(f"OLS log: z_masfem coef={coef:.4f} (~{pct_change:.2f}% change), {sig_text}; direction: {direction}.")
        except Exception as e:
            out["ols_log_fatalities_error"] = str(e)
            descriptions.append(f"OLS log: failed to extract stats ({e}).")
    else:
        descriptions.append("OLS log model not present or is an error message in model_output.")

    # 3) MTurk robustness NB model (z_masfem_mturk)
    if "nb_mturk" in model_output and not isinstance(model_output["nb_mturk"], str):
        try:
            nbm = model_output["nb_mturk"]
            varname = "z_masfem_mturk"
            stats = extract_from_model(nbm, varname)
            irr = np.exp(stats["coef"])
            irr_ci_lower = np.exp(stats["ci_lower"]) if stats["ci_lower"] is not None else None
            irr_ci_upper = np.exp(stats["ci_upper"]) if stats["ci_upper"] is not None else None
            out["nb_mturk"] = {
                "type": "NegativeBinomial (GLM) - MTurk rating",
                "variable": varname,
                **stats,
                "IRR": float(irr),
                "IRR_ci_lower": float(irr_ci_lower) if irr_ci_lower is not None else None,
                "IRR_ci_upper": float(irr_ci_upper) if irr_ci_upper is not None else None,
            }
            p = out["nb_mturk"]["pvalue"]
            coef = out["nb_mturk"]["coef"]
            if p is None:
                sig_text = "p-value not available"
            elif p < 0.01:
                sig_text = "highly statistically significant (p < 0.01)"
            elif p < 0.05:
                sig_text = "statistically significant (p < 0.05)"
            elif p < 0.1:
                sig_text = "marginally significant (p < 0.1)"
            else:
                sig_text = "not statistically significant (p >= 0.1)"
            direction = "associated with more fatalities" if coef > 0 else "associated with fewer fatalities" if coef < 0 else "no association"
            descriptions.append(f"NB MTurk: {varname} coef={coef:.4f}, IRR={irr:.4f}; {sig_text}; direction: {direction}.")
        except Exception as e:
            out["nb_mturk_error"] = str(e)
            descriptions.append(f"NB MTurk: failed to extract stats ({e}).")
    else:
        descriptions.append("NB MTurk model not present or is an error message in model_output.")

    # 4) Means by gender (raw tabulation)
    if "means_by_gender" in model_output:
        try:
            means = model_output["means_by_gender"]
            out["means_by_gender"] = means
            # Provide a short comparison
            male_mean = means.get("mean", {}).get(0, None)
            female_mean = means.get("mean", {}).get(1, None)
            descriptions.append(f"Raw means: male mean fatalities = {male_mean}, female mean fatalities = {female_mean} (see means_by_gender for counts/medians).")
        except Exception as e:
            out["means_by_gender_error"] = str(e)
            descriptions.append(f"Failed to extract means_by_gender ({e}).")

    # Final interpretation summary: focus on primary NB and OLS results if available
    summary_lines = []
    if "nb_primary" in out and isinstance(out["nb_primary"], dict):
        nbp = out["nb_primary"]
        p = nbp["pvalue"]
        coef = nbp["coef"]
        irr = nbp["IRR"]
        if p is None:
            summary_lines.append("Primary NB: effect estimated but p-value unavailable.")
        else:
            if p < 0.05:
                summary_lines.append(f"Primary NB: z_masfem has a statistically significant effect (coef={coef:.4f}, IRR={irr:.3f}, p={p:.3g}).")
            else:
                summary_lines.append(f"Primary NB: z_masfem is not statistically significant (coef={coef:.4f}, IRR={irr:.3f}, p={p:.3g}).")
    if "ols_log_fatalities" in out and isinstance(out["ols_log_fatalities"], dict):
        ols = out["ols_log_fatalities"]
        p = ols["pvalue"]
        coef = ols["coef"]
        pct = ols["pct_change"]
        if p is None:
            summary_lines.append("OLS (log): effect estimated but p-value unavailable.")
        else:
            if p < 0.05:
                summary_lines.append(f"OLS (log): z_masfem has a statistically significant association (coef={coef:.4f}, ~{pct:.2f}% change, p={p:.3g}).")
            else:
                summary_lines.append(f"OLS (log): z_masfem not statistically significant (coef={coef:.4f}, ~{pct:.2f}% change, p={p:.3g}).")

    # If no model stats were extracted, mention that
    if not summary_lines:
        summary_lines.append("No model coefficients were successfully extracted from model_output.")

    description = " ".join(descriptions) + "\n\nSummary interpretation: " + " ".join(summary_lines)

    return {"object": out, "description": description}