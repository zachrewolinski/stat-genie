def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of hurricane name femininity (masfem_z)
    on log-transformed fatalities (primary outcome) from the provided
    statsmodels RegressionResultsWrapper objects.

    Returns a dict with keys:
      - "object": dict with numeric extraction (coefficients, SEs, 95% CIs,
                  p-values, and approximate percent change interpretation)
      - "description": brief textual interpretation about whether results
                       support the hypothesis that more feminine names
                       lead to different fatalities.

    Expects model_output to be the dict returned by the modeling function,
    with keys 'death_model' and optionally 'damage_model'.
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'death_model' and optionally 'damage_model'.")

    if 'death_model' not in model_output or model_output['death_model'] is None:
        raise ValueError("Primary death_model is missing from model_output.")

    death_model = model_output['death_model']

    # Helper to extract effect, se, ci, pvalue for a contrast vector
    def extract_contrast(res, contrast_vec):
        tt = res.t_test(contrast_vec)
        # tt.effect, tt.sd can be arrays; convert to floats
        effect = float(np.atleast_1d(tt.effect)[0])
        se = float(np.atleast_1d(tt.sd)[0]) if hasattr(tt, "sd") else None
        # pvalue may be array or scalar
        pval = float(np.atleast_1d(tt.pvalue)[0])
        ci = tt.conf_int()
        # ci shape (1,2)
        ci_low, ci_high = float(ci[0, 0]), float(ci[0, 1])
        return {"coef": effect, "se": se, "95ci": (ci_low, ci_high), "pvalue": pval}

    # Map param names to indices (robust to ordering)
    params_index = list(death_model.params.index)
    try:
        idx_masfem = params_index.index('masfem_z')
        idx_inter = params_index.index('masfem_x_gender')
    except ValueError as e:
        raise ValueError(f"Expected parameter names 'masfem_z' and 'masfem_x_gender' in model params. Available params: {params_index}") from e

    k = len(params_index)
    zero_vec = np.zeros(k)

    # Effect for male-coded names (gender_mf = 0): just masfem_z coefficient
    vec_male = zero_vec.copy()
    vec_male[idx_masfem] = 1.0
    male_res = extract_contrast(death_model, vec_male)

    # Effect for female-coded names (gender_mf = 1): masfem_z + masfem_x_gender
    vec_female = zero_vec.copy()
    vec_female[idx_masfem] = 1.0
    vec_female[idx_inter] = 1.0
    female_res = extract_contrast(death_model, vec_female)

    # Convert coefficients on log(y+1) to approximate percent change in (y+1)
    def pct_change_from_logcoef(coef):
        try:
            return (np.exp(coef) - 1.0) * 100.0
        except Exception:
            return None

    male_res["pct_change_approx"] = pct_change_from_logcoef(male_res["coef"])
    female_res["pct_change_approx"] = pct_change_from_logcoef(female_res["coef"])

    output = {
        "death_model": {
            "params_order": params_index,
            "male_coded_names_effect_per_SD_masfem": male_res,
            "female_coded_names_effect_per_SD_masfem": female_res,
            "notes": "Effects are per 1 SD increase in masfem_z. Outcome is log(alldeaths + 1). "
                     "Approx percent change reported is (exp(coef)-1)*100, i.e., approximate percent change in (deaths+1)."
        }
    }

    # If damage_model present and non-null, extract same diagnostics for sensitivity
    if 'damage_model' in model_output and model_output['damage_model'] is not None:
        damage_model = model_output['damage_model']
        params_index_d = list(damage_model.params.index)
        try:
            idx_masfem_d = params_index_d.index('masfem_z')
            idx_inter_d = params_index_d.index('masfem_x_gender')
            kd = len(params_index_d)
            zero_vec_d = np.zeros(kd)
            vec_male_d = zero_vec_d.copy(); vec_male_d[idx_masfem_d] = 1.0
            vec_female_d = zero_vec_d.copy(); vec_female_d[idx_masfem_d] = 1.0; vec_female_d[idx_inter_d] = 1.0
            male_d = extract_contrast(damage_model, vec_male_d)
            female_d = extract_contrast(damage_model, vec_female_d)
            male_d["pct_change_approx"] = pct_change_from_logcoef(male_d["coef"])
            female_d["pct_change_approx"] = pct_change_from_logcoef(female_d["coef"])

            output["damage_model"] = {
                "params_order": params_index_d,
                "male_coded_names_effect_per_SD_masfem": male_d,
                "female_coded_names_effect_per_SD_masfem": female_d,
                "notes": "Secondary outcome: log(ndam15 + 1). Same interpretation procedure as for fatalities."
            }
        except ValueError:
            # If expected param names missing, include a note
            output["damage_model"] = {"error": "damage_model present but expected parameter names not found.",
                                      "available_params": params_index_d}

    # Formulate a concise description / conclusion about the hypothesis
    # Hypothesis: more feminine names -> fewer precautions -> more fatalities
    # Therefore we look for a positive and statistically significant coefficient for femininity (especially among female-coded names).
    concl_lines = []
    # Primary test focuses on female-coded names effect (where the interaction applies)
    fe = female_res["coef"]
    fp = female_res["pvalue"]
    if fp < 0.05 and fe > 0:
        concl_lines.append(
            f"Primary result: For hurricanes with female-coded names, a 1 SD increase in name femininity "
            f"is associated with an increase in log(deaths+1) of {fe:.3f} (95% CI [{female_res['95ci'][0]:.3f}, {female_res['95ci'][1]:.3f}], p = {fp:.3g}), "
            f"≈ {female_res['pct_change_approx']:.1f}% increase in (deaths+1). This is consistent with the hypothesis."
        )
    elif fp < 0.05 and fe < 0:
        concl_lines.append(
            f"Primary result: For female-coded names, a 1 SD increase in name femininity is associated with a DECREASE in log(deaths+1) "
            f"({fe:.3f}, 95% CI [{female_res['95ci'][0]:.3f}, {female_res['95ci'][1]:.3f}], p = {fp:.3g}), "
            f"≈ {female_res['pct_change_approx']:.1f}% change. This is opposite the hypothesis."
        )
    else:
        concl_lines.append(
            f"Primary result: For female-coded names, the estimated effect of a 1 SD increase in name femininity on log(deaths+1) is "
            f"{fe:.3f} (95% CI [{female_res['95ci'][0]:.3f}, {female_res['95ci'][1]:.3f}], p = {fp:.3g}). "
            f"This is not statistically significant at p<0.05, so there is no strong evidence supporting the hypothesis in the primary model."
        )

    # Optionally mention male-coded names
    me = male_res["coef"]
    mp = male_res["pvalue"]
    concl_lines.append(
        f"For male-coded names, the effect per 1 SD feminine rating is {me:.3f} (95% CI [{male_res['95ci'][0]:.3f}, {male_res['95ci'][1]:.3f}], p = {mp:.3g}), "
        f"≈ {male_res['pct_change_approx']:.1f}% change in (deaths+1)."
    )

    # Mention damage sensitivity if available
    if "damage_model" in output and isinstance(output["damage_model"], dict) and "male_coded_names_effect_per_SD_masfem" in output["damage_model"]:
        fd = output["damage_model"]["female_coded_names_effect_per_SD_masfem"]
        fp_d = fd["pvalue"]
        fe_d = fd["coef"]
        concl_lines.append(
            f"Sensitivity (damages): For female-coded names the coefficient is {fe_d:.3f} (95% CI [{fd['95ci'][0]:.3f}, {fd['95ci'][1]:.3f}], p = {fp_d:.3g}), "
            f"≈ {fd['pct_change_approx']:.1f}% change in (damages+1)."
        )

    description = " ".join(concl_lines)

    return {"object": output, "description": description}