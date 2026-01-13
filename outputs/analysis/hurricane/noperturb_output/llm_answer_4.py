def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, 95% CI, and exponentiated effect for the 'female_name'
    variable (and masfem_z as a continuous counterpart) from the provided model_output dict.
    Also extracts interaction term from the interaction model and the female_name effect from
    the damage model as a robustness check.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Concise interpretation of results in context"
      }
    """
    import numpy as np

    out = {}
    notes = []

    def _extract_from_result(res, term):
        """Return dict with coef, se, pvalue, ci_lower, ci_upper, exp_coef if term exists, else None."""
        if res is None:
            return None
        params = getattr(res, "params", None)
        if params is None:
            return None
        if term not in params.index:
            return None
        coef = float(params[term])
        se = float(res.bse[term]) if hasattr(res, "bse") else None
        pval = float(res.pvalues[term]) if hasattr(res, "pvalues") else None
        ci = res.conf_int(alpha=0.05).loc[term].tolist() if hasattr(res, "conf_int") else [None, None]
        exp_coef = float(np.exp(coef)) if coef is not None else None
        return {
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": float(ci[0]) if ci[0] is not None else None,
            "ci_upper": float(ci[1]) if ci[1] is not None else None,
            "exp_coef": exp_coef
        }

    # Retrieve models from dict (safe get)
    main = model_output.get("ols_log_deaths_main")
    interact = model_output.get("ols_log_deaths_interact")
    damage = model_output.get("ols_log_damage")

    # Primary term: female_name in main model
    main_female = _extract_from_result(main, "female_name")
    if main_female is not None:
        out["main_female_name"] = main_female
        sig = main_female["pvalue"] < 0.05 if main_female["pvalue"] is not None else None
        notes.append(
            "Main model (OLS on log(alldeaths+1)): female_name coef = {coef:.4f} (SE={se:.4f}), p = {p:.4g}, "
            "95% CI [{lo:.4f}, {hi:.4f}]. Exponentiated effect on (alldeaths+1): {exp:.3f}x. {sig}".format(
                coef=main_female["coef"], se=main_female["se"], p=main_female["pvalue"],
                lo=main_female["ci_lower"], hi=main_female["ci_upper"], exp=main_female["exp_coef"],
                sig=("Statistically significant at alpha=0.05." if sig else "Not statistically significant.")
            )
        )
    else:
        notes.append("Main model: 'female_name' term not found in model output.")

    # Continuous masculinity-femininity (masfem_z) in main model
    main_masfem = _extract_from_result(main, "masfem_z")
    if main_masfem is not None:
        out["main_masfem_z"] = main_masfem
        sig = main_masfem["pvalue"] < 0.05 if main_masfem["pvalue"] is not None else None
        notes.append(
            "Main model: masfem_z coef = {coef:.4f} (SE={se:.4f}), p = {p:.4g}, 95% CI [{lo:.4f}, {hi:.4f}]. Exponentiated effect: {exp:.3f}x. {sig}".format(
                coef=main_masfem["coef"], se=main_masfem["se"], p=main_masfem["pvalue"],
                lo=main_masfem["ci_lower"], hi=main_masfem["ci_upper"], exp=main_masfem["exp_coef"],
                sig=("Significant." if sig else "Not significant.")
            )
        )
    else:
        notes.append("Main model: 'masfem_z' term not found in model output.")

    # Interaction model: female_name main effect and interaction term with storm_severity
    interact_female = _extract_from_result(interact, "female_name")
    # Possible interaction term name: 'female_name:storm_severity' (statsmodels uses colon)
    interaction_term_name = "female_name:storm_severity"
    interact_inter = _extract_from_result(interact, interaction_term_name)
    if interact_female is not None:
        out["interact_female_name"] = interact_female
        notes.append("Interaction model: female_name main effect extracted.")
    else:
        notes.append("Interaction model: female_name main effect not found.")

    if interact_inter is not None:
        out["female_by_stormseverity_interaction"] = interact_inter
        sig = interact_inter["pvalue"] < 0.05 if interact_inter["pvalue"] is not None else None
        notes.append(
            "Interaction term (female_name:storm_severity) coef = {coef:.4f} (SE={se:.4f}), p = {p:.4g}, 95% CI [{lo:.4f}, {hi:.4f}]. {sig}".format(
                coef=interact_inter["coef"], se=interact_inter["se"], p=interact_inter["pvalue"],
                lo=interact_inter["ci_lower"], hi=interact_inter["ci_upper"],
                sig=("Significant interaction." if sig else "No significant interaction.")
            )
        )
    else:
        notes.append("Interaction model: female_name:storm_severity term not found.")

    # Damage model: female_name as robustness outcome
    damage_female = _extract_from_result(damage, "female_name")
    if damage_female is not None:
        out["damage_female_name"] = damage_female
        sig = damage_female["pvalue"] < 0.05 if damage_female["pvalue"] is not None else None
        notes.append(
            "Damage model (log damages): female_name coef = {coef:.4f} (SE={se:.4f}), p = {p:.4g}, 95% CI [{lo:.4f}, {hi:.4f}]. Exponentiated effect: {exp:.3f}x. {sig}".format(
                coef=damage_female["coef"], se=damage_female["se"], p=damage_female["pvalue"],
                lo=damage_female["ci_lower"], hi=damage_female["ci_upper"], exp=damage_female["exp_coef"],
                sig=("Significant." if sig else "Not significant.")
            )
        )
    else:
        notes.append("Damage model: 'female_name' term not found.")

    # Final concise interpretation about the hypothesis:
    # Use p-values from main model female_name and masfem_z if available to form a short conclusion.
    conclusion = "Could not form conclusion: relevant statistics missing."
    if main_female is not None:
        if main_female["pvalue"] is not None:
            if main_female["pvalue"] < 0.05:
                if main_female["coef"] > 0:
                    conclusion = ("Primary result: female-named hurricanes are associated with higher log fatalities "
                                  "in the main model (coef={:.4f}, p={:.4g}), consistent with the hypothesis that "
                                  "feminine names lead to fewer precautions and higher deaths.").format(
                                      main_female["coef"], main_female["pvalue"])
                else:
                    conclusion = ("Primary result: female-named hurricanes are associated with LOWER log fatalities "
                                  "in the main model (coef={:.4f}, p={:.4g}), contrary to the hypothesis.").format(
                                      main_female["coef"], main_female["pvalue"])
            else:
                conclusion = ("Primary result: no statistically significant association between female_name and "
                              "log fatalities in the main model (coef={:.4f}, p={:.4g}); this does not support the "
                              "hypothesis.").format(main_female["coef"], main_female["pvalue"])
        else:
            conclusion = "Primary result: female_name estimate available but p-value missing; cannot assess significance."
    elif main_masfem is not None:
        # fallback to continuous measure
        if main_masfem["pvalue"] is not None:
            if main_masfem["pvalue"] < 0.05:
                direction = "higher" if main_masfem["coef"] > 0 else "lower"
                conclusion = ("Primary result (masfem_z): more feminine names are associated with {} log fatalities "
                              "(coef={:.4f}, p={:.4g}), {} the hypothesis.").format(direction, main_masfem["coef"], main_masfem["pvalue"])
            else:
                conclusion = ("Primary result (masfem_z): no statistically significant association between name femininity "
                              "and log fatalities (coef={:.4f}, p={:.4g}).").format(main_masfem["coef"], main_masfem["pvalue"])
        else:
            conclusion = "Primary result: masfem_z estimate available but p-value missing; cannot assess significance."
    else:
        conclusion = "Primary model terms for female_name and masfem_z not found; cannot form conclusion."

    # Aggregate description
    description = " ; ".join(notes) + " || Conclusion: " + conclusion

    return {"object": out, "description": description}