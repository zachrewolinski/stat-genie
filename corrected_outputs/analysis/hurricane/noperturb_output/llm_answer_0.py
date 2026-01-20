def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted models to answer whether more-feminine
    hurricane names predict larger damages.

    Input:
      model_output: dict-like object expected to contain at least the key
                    'damage_model' mapped to a statsmodels RegressionResultsWrapper.

    Returns:
      dict with keys:
        - "object": dict with extracted numeric statistics for 'masfem_z' and
                    'gender_female' (if present), plus model N and R-squared.
        - "description": short plain-language interpretation of the results
                         in the context of the hypothesis.
    """
    import numpy as np

    out = {"object": {}, "description": ""}

    if not isinstance(model_output, dict):
        out["description"] = "model_output is not a dict; expected dict with 'damage_model'."
        return out

    damage_model = model_output.get("damage_model", None)
    if damage_model is None:
        out["description"] = "No damage_model found in model_output."
        return out

    # Helper to extract stats for a given variable name if present in the model
    def extract_for(varname):
        res = {}
        params = getattr(damage_model, "params", None)
        if params is None or varname not in params.index:
            return None
        beta = float(params[varname])
        bse = float(damage_model.bse[varname])
        t = float(damage_model.tvalues[varname])
        p = float(damage_model.pvalues[varname])
        ci_low, ci_high = [float(x) for x in damage_model.conf_int().loc[varname].values]
        # Convert log outcome effect to percent change in original ndam15 scale:
        # percent_change = (exp(beta) - 1) * 100
        pct_change = (np.exp(beta) - 1.0) * 100.0
        res.update({
            "coef": beta,
            "std_err": bse,
            "t": t,
            "p_value": p,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "percent_change": pct_change,  # percent change in ndam15+1 per unit increase
        })
        return res

    # Extract for masfem_z (primary continuous femininity score)
    masfem_stats = extract_for("masfem_z")
    gender_stats = extract_for("gender_female")

    # Model-level info
    try:
        nobs = int(damage_model.nobs)
    except Exception:
        nobs = None
    try:
        rsq = float(getattr(damage_model, "rsquared", np.nan))
    except Exception:
        rsq = None

    out["object"]["masfem_z"] = masfem_stats
    out["object"]["gender_female"] = gender_stats
    out["object"]["nobs"] = nobs
    out["object"]["rsquared"] = rsq

    # Build interpretation
    lines = []
    lines.append(f"Model used log_ndam15 as outcome; coefficients represent change in log(ndam15+1).")
    lines.append(f"Model N = {nobs}, R^2 = {rsq:.3f}" if rsq is not None else f"Model N = {nobs}")

    if masfem_stats is not None:
        lines.append(
            "masfem_z: coef = {coef:.4f}, SE = {std_err:.4f}, t = {t:.2f}, p = {p_value:.3g}, "
            "95% CI = [{ci_95_lower:.4f}, {ci_95_upper:.4f}]. "
            "This implies a {percent_change:.2f}% change in ndam15 per 1 SD increase in femininity."
            .format(**masfem_stats)
        )
        if masfem_stats["p_value"] < 0.05:
            if masfem_stats["coef"] > 0:
                lines.append("Interpretation: The positive and statistically significant coefficient indicates that more-feminine names are associated with higher damages, which is consistent with the hypothesis (feminine names -> fewer precautions -> larger damages).")
            else:
                lines.append("Interpretation: The negative and statistically significant coefficient indicates that more-feminine names are associated with lower damages (opposite of the hypothesis).")
        else:
            lines.append("Interpretation: The coefficient is not statistically significant at conventional levels (p >= 0.05); there is no strong evidence that the continuous femininity score predicts damages.")
    else:
        lines.append("masfem_z is not present in the fitted damage model.")

    if gender_stats is not None:
        lines.append(
            "gender_female (binary): coef = {coef:.4f}, SE = {std_err:.4f}, t = {t:.2f}, p = {p_value:.3g}, "
            "95% CI = [{ci_95_lower:.4f}, {ci_95_upper:.4f}]. "
            "This implies a {percent_change:.2f}% difference in ndam15 for female- vs male-coded names."
            .format(**gender_stats)
        )
        if gender_stats["p_value"] < 0.05:
            if gender_stats["coef"] > 0:
                lines.append("Interpretation: Female-coded names are associated with significantly higher damages compared to male-coded names, consistent with the hypothesis.")
            else:
                lines.append("Interpretation: Female-coded names are associated with significantly lower damages compared to male-coded names (opposite of the hypothesis).")
        else:
            lines.append("Interpretation: The binary gender indicator is not statistically significant at conventional levels (p >= 0.05); no strong evidence for a female/male difference in damages.")
    else:
        lines.append("gender_female is not present in the fitted damage model.")

    # Final verdict summary
    # If either coefficient is positive and significant -> support; if negative and significant -> refute; otherwise inconclusive
    verdict = "Inconclusive based on these coefficients."
    pos_sig = (masfem_stats and masfem_stats["p_value"] < 0.05 and masfem_stats["coef"] > 0) or \
              (gender_stats and gender_stats["p_value"] < 0.05 and gender_stats["coef"] > 0)
    neg_sig = (masfem_stats and masfem_stats["p_value"] < 0.05 and masfem_stats["coef"] < 0) or \
              (gender_stats and gender_stats["p_value"] < 0.05 and gender_stats["coef"] < 0)
    if pos_sig and not neg_sig:
        verdict = "Overall: Evidence supports the hypothesis that more-feminine names are associated with larger damages (consistent with fewer precautions)."
    elif neg_sig and not pos_sig:
        verdict = "Overall: Evidence runs counter to the hypothesis (feminine names associated with lower damages)."
    elif pos_sig and neg_sig:
        verdict = "Overall: Mixed significant results (some coefficients point in opposite directions); no clear single conclusion."
    else:
        verdict = "Overall: No statistically significant evidence supporting the hypothesis in this model."

    lines.append(verdict)
    out["description"] = " ".join(lines)

    return out