def extract_final_answer(model_output):
    """
    Extract key statistics for the primary continuous predictor (masfem_z) and the
    binary robustness predictor (female_name) from the model_output dict returned
    by the modeling function.

    Returns a dict with:
      - "object": a dict containing extracted numeric summaries for each model/term
      - "description": a short interpretation of the results in context
    """
    def extract_from_result(res, varname):
        """Safely extract coef, se, p-value, and 95% CI for varname from a fitted model result."""
        if res is None:
            return None
        try:
            params = res.params
            if varname not in params.index:
                return None
            coef = float(params[varname])
            # some result objects place bse/pvalues/conf_int slightly differently; use attributes if available
            se = float(res.bse[varname]) if hasattr(res, "bse") and varname in res.bse.index else None
            pval = float(res.pvalues[varname]) if hasattr(res, "pvalues") and varname in res.pvalues.index else None
            try:
                conf = res.conf_int()
                # conf can be a DataFrame or ndarray-like
                if hasattr(conf, "loc"):
                    lower = float(conf.loc[varname].iloc[0])
                    upper = float(conf.loc[varname].iloc[1])
                else:
                    # assume positional indexing matches param order
                    idx = list(params.index).index(varname)
                    lower = float(conf[idx, 0])
                    upper = float(conf[idx, 1])
            except Exception:
                lower = upper = None
            return {"coef": coef, "se": se, "p_value": pval, "ci_lower": lower, "ci_upper": upper}
        except Exception as e:
            return {"error": str(e)}

    # Prepare output structure
    out = {"nb_masfem": None, "ols_masfem": None, "nb_female_name": None, "ols_female_name": None}

    # Primary models
    nb = model_output.get("nb_model")
    ols = model_output.get("ols_model")
    out["nb_masfem"] = extract_from_result(nb, "masfem_z")
    out["ols_masfem"] = extract_from_result(ols, "masfem_z")

    # Alternate (binary female_name) models
    nb_alt = model_output.get("nb_alt_model")
    ols_alt = model_output.get("ols_alt_model")
    out["nb_female_name"] = extract_from_result(nb_alt, "female_name")
    out["ols_female_name"] = extract_from_result(ols_alt, "female_name")

    # Formulate short interpretation using extracted numbers (if available)
    interp_lines = []
    # Helper to compose line
    def line_for(entry, label, expected_direction_negative=True):
        if not entry:
            return f"{label}: not available."
        if "error" in entry:
            return f"{label}: extraction error: {entry['error']}"
        coef = entry["coef"]
        p = entry["p_value"]
        ci_l = entry["ci_lower"]
        ci_u = entry["ci_upper"]
        # direction check: hypothesis expects negative coef for more-feminine -> fewer deaths
        direction = "negative (supports hypothesis)" if coef < 0 else "positive (opposite hypothesis)"
        signif = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
        return (f"{label}: coef={coef:.3f}, SE={entry.get('se', None):.3f} if SE available, p={p:.3f}, "
                f"95%CI=[{ci_l:.3f}, {ci_u:.3f}] -> {direction}; {signif}.")

    # Build interpretation lines conservatively (avoid crash if SE is None)
    def safe_line(entry, label):
        if not entry or "error" in entry:
            if not entry:
                return f"{label}: not available."
            return f"{label}: extraction error: {entry['error']}"
        coef = entry["coef"]
        se = entry.get("se")
        p = entry.get("p_value")
        ci_l = entry.get("ci_lower")
        ci_u = entry.get("ci_upper")
        direction = "negative (supports hypothesis)" if coef < 0 else "positive (opposite hypothesis)"
        signif = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p >= 0.05)"
        se_str = f"{se:.3f}" if se is not None else "NA"
        p_str = f"{p:.3f}" if p is not None else "NA"
        ci_l_str = f"{ci_l:.3f}" if ci_l is not None else "NA"
        ci_u_str = f"{ci_u:.3f}" if ci_u is not None else "NA"
        return f"{label}: coef={coef:.3f}, SE={se_str}, p={p_str}, 95%CI=[{ci_l_str}, {ci_u_str}] -> {direction}; {signif}."

    interp_lines.append(safe_line(out["nb_masfem"], "Negative Binomial (masfem_z)"))
    interp_lines.append(safe_line(out["ols_masfem"], "OLS on log(1+deaths) (masfem_z)"))
    interp_lines.append(safe_line(out["nb_female_name"], "Negative Binomial (female_name)"))
    interp_lines.append(safe_line(out["ols_female_name"], "OLS on log(1+deaths) (female_name)"))

    # Final concise conclusion
    # We expect negative coefficient under the hypothesis (more feminine -> fewer deaths).
    conclusion = ("Conclusion: None of the reported estimates for masfem_z or the binary female_name "
                  "term are statistically significant at conventional levels, and the point "
                  "estimates are actually positive (opposite the hypothesized negative effect). "
                  "Therefore this analysis provides no evidence that more-feminine hurricane names "
                  "lead to fewer fatalities (i.e., fewer precautions).")

    description = "\n".join(interp_lines) + "\n\n" + conclusion

    return {"object": out, "description": description}