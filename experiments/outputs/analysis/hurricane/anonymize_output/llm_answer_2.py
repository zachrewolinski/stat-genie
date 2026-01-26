def extract_final_answer(model_output):
    """
    Extract key statistics from model_output dict produced by the provided modeling function.
    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results for the primary hypothesis (MasFem_z),
                  auxiliary binary indicator (IsFemaleName), MTurk-based measure (if present),
                  and the damage-auxiliary model.
      - "description": a concise interpretation of those estimates in the context of the hypothesis.
    """
    import math
    results = {}
    desc_lines = []
    
    def summarize_param(model, param_name):
        """Return a summary dict for param_name in given statsmodels result or None if missing."""
        if model is None:
            return None
        params = getattr(model, "params", None)
        if params is None or param_name not in params.index:
            return None
        coef = float(params.loc[param_name])
        pval = float(model.pvalues.loc[param_name])
        tval = float(model.tvalues.loc[param_name]) if hasattr(model, "tvalues") else None
        try:
            ci = model.conf_int().loc[param_name].astype(float).tolist()
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            ci_low, ci_high = None, None
        # convert log outcome coefficient to multiplicative % change:
        try:
            pct_change = (math.exp(coef) - 1.0) * 100.0
            ci_low_pct = (math.exp(ci_low) - 1.0) * 100.0 if ci_low is not None else None
            ci_high_pct = (math.exp(ci_high) - 1.0) * 100.0 if ci_high is not None else None
        except Exception:
            pct_change, ci_low_pct, ci_high_pct = None, None, None
        summary = {
            "coef": coef,
            "p_value": pval,
            "t_value": tval,
            "ci_95": [ci_low, ci_high],
            "percent_change_outcome": pct_change,
            "percent_change_ci_95": [ci_low_pct, ci_high_pct],
        }
        return summary

    # Helper to safely get model by key
    def get_model(key):
        return model_output.get(key, None) if isinstance(model_output, dict) else None

    mod_deaths = get_model("model_deaths")
    mod_damage = get_model("model_damage")
    mod_deaths_mturk = get_model("model_deaths_mturk")

    # Primary: MasFem_z effect on LogDeaths
    mas_summary = summarize_param(mod_deaths, "MasFem_z")
    results["MasFem_z_on_LogDeaths"] = mas_summary

    # Auxiliary: IsFemaleName effect on LogDeaths
    female_summary = summarize_param(mod_deaths, "IsFemaleName")
    results["IsFemaleName_on_LogDeaths"] = female_summary

    # MTurk robustness: MTurk_MasFem_z on LogDeaths
    mturk_summary = summarize_param(mod_deaths_mturk, "MTurk_MasFem_z") if mod_deaths_mturk is not None else None
    results["MTurk_MasFem_z_on_LogDeaths"] = mturk_summary

    # Auxiliary outcome: MasFem_z effect on LogDamage2013
    mas_damage_summary = summarize_param(mod_damage, "MasFem_z")
    results["MasFem_z_on_LogDamage2013"] = mas_damage_summary

    # Add sample size and R-squared information for the main model if available
    if mod_deaths is not None:
        try:
            results["model_deaths_nobs"] = int(mod_deaths.nobs)
            results["model_deaths_r2"] = float(getattr(mod_deaths, "rsquared", float("nan")))
        except Exception:
            pass
    if mod_damage is not None:
        try:
            results["model_damage_nobs"] = int(mod_damage.nobs)
            results["model_damage_r2"] = float(getattr(mod_damage, "rsquared", float("nan")))
        except Exception:
            pass
    if mod_deaths_mturk is not None:
        try:
            results["model_deaths_mturk_nobs"] = int(mod_deaths_mturk.nobs)
            results["model_deaths_mturk_r2"] = float(getattr(mod_deaths_mturk, "rsquared", float("nan")))
        except Exception:
            pass

    # Build a short interpretation
    if mas_summary is None:
        desc_lines.append("MasFem_z coefficient not found in model_deaths.")
    else:
        coef = mas_summary["coef"]
        p = mas_summary["p_value"]
        pct = mas_summary["percent_change_outcome"]
        ci_pct = mas_summary["percent_change_ci_95"]
        sign = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        sig = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p >= 0.05)" 
        desc_lines.append(
            f"Primary result: MasFem_z -> LogDeaths: coefficient = {coef:.4f}, p = {p:.3g}. "
            f"This is {sign} and {sig}. "
            + (f"Interpreted on the deaths scale: ≈ {pct:.1f}% change in expected fatalities per 1-SD increase in name femininity "
               f"(95% CI ≈ [{ci_pct[0]:.1f}%, {ci_pct[1]:.1f}%])."
               if (pct is not None and ci_pct[0] is not None) else "")
        )

    if female_summary is not None:
        coef = female_summary["coef"]
        p = female_summary["p_value"]
        pct = female_summary["percent_change_outcome"]
        desc_lines.append(
            f"Auxiliary: IsFemaleName -> LogDeaths: coefficient = {coef:.4f}, p = {p:.3g}. "
            + (f"Approx {pct:.1f}% change in deaths for female vs male names."
               if pct is not None else "")
        )
    else:
        desc_lines.append("Auxiliary IsFemaleName coefficient not found in model_deaths.")

    if mturk_summary is not None:
        coef = mturk_summary["coef"]
        p = mturk_summary["p_value"]
        pct = mturk_summary["percent_change_outcome"]
        desc_lines.append(
            f"Robustness (MTurk measure): MTurk_MasFem_z -> LogDeaths: coefficient = {coef:.4f}, p = {p:.3g}. "
            + (f"Approx {pct:.1f}% change per 1-SD increase in MTurk femininity rating."
               if pct is not None else "")
        )
    else:
        desc_lines.append("MTurk-based model not available or MTurk_MasFem_z missing.")

    if mas_damage_summary is not None:
        coef = mas_damage_summary["coef"]
        p = mas_damage_summary["p_value"]
        pct = mas_damage_summary["percent_change_outcome"]
        desc_lines.append(
            f"Auxiliary outcome (damage): MasFem_z -> LogDamage2013: coefficient = {coef:.4f}, p = {p:.3g}. "
            + (f"Approx {pct:.1f}% change in logged damage per 1-SD increase."
               if pct is not None else "")
        )
    else:
        desc_lines.append("MasFem_z coefficient not found in model_damage (auxiliary damage model).")

    # Short overall conclusion sentence about hypothesis
    # Determine direction and significance of primary effect to conclude yes/no
    conclusion = "Unable to determine (missing results)."
    if mas_summary is not None:
        coef = mas_summary["coef"]
        p = mas_summary["p_value"]
        if coef > 0 and p < 0.05:
            conclusion = "Evidence consistent with the hypothesis: more feminine names are associated with higher fatalities (fewer precautions)."
        elif coef > 0 and p >= 0.05:
            conclusion = "Point estimate is in the hypothesized direction (more feminine -> higher fatalities) but it is not statistically significant."
        elif coef < 0 and p < 0.05:
            conclusion = "Evidence contradicts the hypothesis: more feminine names are associated with lower fatalities (statistically significant)."
        elif coef < 0 and p >= 0.05:
            conclusion = "Point estimate is opposite the hypothesized direction (more feminine -> fewer fatalities) but it is not statistically significant."
        else:
            conclusion = "No clear effect."

    desc_lines.append("Conclusion: " + conclusion)

    return {"object": results, "description": " ".join(desc_lines)}