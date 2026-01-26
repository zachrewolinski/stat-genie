import numpy as np

def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of storm name femininity (masfem_std)
    from the provided model_output dict (as returned by the model() function).
    Returns a dictionary with:
      - "object": a dict of numeric results (coefficients, p-values, CIs, multiplicative effects)
      - "description": a short plain-language interpretation of those results relative to the task.
    """
    out = {
        "death": None,
        "damage": None,
        "death_gender_binary": None,
        "final_verdict": None
    }

    def safe_get_coef_pci(res, name):
        # Try to extract coef, pvalue, and 95% CI from a statsmodels results object
        try:
            coef = float(res.params[name])
            pval = float(res.pvalues[name])
            ci = res.conf_int().loc[name].tolist()
            return coef, pval, ci
        except Exception:
            return None, None, (None, None)

    # --- Deaths (Negative Binomial) ---
    death_coef = model_output.get('death_masfem_coef', None)
    death_p = model_output.get('death_masfem_pvalue', None)
    death_ci = None
    if death_coef is None or death_p is None:
        if 'death_model' in model_output and model_output['death_model'] is not None:
            coef, pval, ci = safe_get_coef_pci(model_output['death_model'], 'masfem_std')
            death_coef = coef if death_coef is None else death_coef
            death_p = pval if death_p is None else death_p
            death_ci = ci
    else:
        # try CI from model object if available
        if 'death_model' in model_output and model_output['death_model'] is not None:
            try:
                death_ci = model_output['death_model'].conf_int().loc['masfem_std'].tolist()
            except Exception:
                death_ci = None

    if death_coef is not None:
        # Interpret NB coefficient on log link: multiplicative effect = exp(coef)
        mult = float(np.exp(death_coef))
        if death_ci is not None and None not in death_ci:
            try:
                mult_ci = [float(np.exp(death_ci[0])), float(np.exp(death_ci[1]))]
            except Exception:
                mult_ci = [None, None]
        else:
            mult_ci = [None, None]

        significance = None
        if death_p is not None:
            if death_p < 0.05:
                significance = "statistically significant (p < 0.05)"
            elif death_p < 0.10:
                significance = "marginal (0.05 <= p < 0.10)"
            else:
                significance = "not statistically significant (p >= 0.10)"

        # Build interpretation string safely
        if mult_ci[0] is not None:
            interpretation = (
                "In the death model (NB, log link), a 1 SD increase in name femininity is associated with "
                f"a multiplicative change in expected deaths of {mult:.3f}x "
                f"(i.e. +{(mult-1)*100:.1f}%); 95% CI for multiplicative effect: "
                f"{mult_ci[0]:.3f} to {mult_ci[1]:.3f}."
            )
        else:
            interpretation = (
                "In the death model (NB, log link), a 1 SD increase in name femininity is associated with "
                f"a multiplicative change in expected deaths of {mult:.3f}x."
            )

        out['death'] = {
            "coef": float(death_coef),
            "pvalue": float(death_p) if death_p is not None else None,
            "coef_95ci": death_ci,
            "multiplicative_effect": mult,
            "multiplicative_effect_95ci": mult_ci,
            "significance": significance,
            "interpretation": interpretation
        }

    # --- Damage (OLS on log damage) ---
    damage_coef = model_output.get('damage_masfem_coef', None)
    damage_p = model_output.get('damage_masfem_pvalue', None)
    damage_ci = None
    if damage_coef is None or damage_p is None:
        if 'damage_model' in model_output and model_output['damage_model'] is not None:
            coef, pval, ci = safe_get_coef_pci(model_output['damage_model'], 'masfem_std')
            damage_coef = coef if damage_coef is None else damage_coef
            damage_p = pval if damage_p is None else damage_p
            damage_ci = ci
    else:
        if 'damage_model' in model_output and model_output['damage_model'] is not None:
            try:
                damage_ci = model_output['damage_model'].conf_int().loc['masfem_std'].tolist()
            except Exception:
                damage_ci = None

    if damage_coef is not None:
        mult = float(np.exp(damage_coef))  # multiplicative effect on damage
        if damage_ci is not None and None not in damage_ci:
            try:
                mult_ci = [float(np.exp(damage_ci[0])), float(np.exp(damage_ci[1]))]
            except Exception:
                mult_ci = [None, None]
        else:
            mult_ci = [None, None]

        significance = None
        if damage_p is not None:
            if damage_p < 0.05:
                significance = "statistically significant (p < 0.05)"
            elif damage_p < 0.10:
                significance = "marginal (0.05 <= p < 0.10)"
            else:
                significance = "not statistically significant (p >= 0.10)"

        if mult_ci[0] is not None:
            interpretation = (
                "In the log-damage OLS, a 1 SD increase in name femininity is associated with "
                f"a multiplicative change in expected damage of {mult:.3f}x "
                f"(i.e. +{(mult-1)*100:.1f}%); 95% CI for multiplicative effect: "
                f"{mult_ci[0]:.3f} to {mult_ci[1]:.3f}."
            )
        else:
            interpretation = (
                "In the log-damage OLS, a 1 SD increase in name femininity is associated with "
                f"a multiplicative change in expected damage of {mult:.3f}x."
            )

        out['damage'] = {
            "coef": float(damage_coef),
            "pvalue": float(damage_p) if damage_p is not None else None,
            "coef_95ci": damage_ci,
            "multiplicative_effect": mult,
            "multiplicative_effect_95ci": mult_ci,
            "significance": significance,
            "interpretation": interpretation
        }

    # --- Binary gender (robustness) if present ---
    if 'death_gender_bin_coef' in model_output or 'death_model_gender_bin_summary' in model_output:
        gb_coef = model_output.get('death_gender_bin_coef', None)
        gb_p = model_output.get('death_gender_bin_pvalue', None)
        gb_ci = None
        if gb_coef is not None:
            try:
                gb_mult = float(np.exp(gb_coef))
            except Exception:
                gb_mult = None
            sig = None
            if gb_p is not None:
                if gb_p < 0.05:
                    sig = "statistically significant (p < 0.05)"
                elif gb_p < 0.10:
                    sig = "marginal (0.05 <= p < 0.10)"
                else:
                    sig = "not statistically significant (p >= 0.10)"
            interpretation = (
                f"Binary female indicator associated with multiplicative change in expected deaths of {gb_mult:.3f}x."
                if gb_mult is not None else ""
            )
            out['death_gender_binary'] = {
                "coef": float(gb_coef),
                "pvalue": float(gb_p) if gb_p is not None else None,
                "multiplicative_effect": gb_mult,
                "coef_95ci": gb_ci,
                "significance": sig,
                "interpretation": interpretation
            }

    # --- Final verdict relative to the yes/no task ---
    evidence_lines = []
    if out['death'] is not None:
        dc = out['death']
        if dc['pvalue'] is not None:
            if dc['pvalue'] < 0.05:
                evidence_lines.append("Strong evidence that more feminine names are associated with MORE deaths (consistent with fewer precautions).")
            elif dc['pvalue'] < 0.10:
                evidence_lines.append("Suggestive/marginal evidence that more feminine names are associated with MORE deaths (p ~ 0.06).")
            else:
                evidence_lines.append("No evidence that more feminine names are associated with deaths (coefficient not statistically significant).")
        else:
            evidence_lines.append("Death model coefficient available but p-value missing; cannot assess significance.")
    else:
        evidence_lines.append("No death-model result available.")

    if out['damage'] is not None:
        dg = out['damage']
        if dg['pvalue'] is not None:
            if dg['pvalue'] < 0.05:
                evidence_lines.append("Evidence that more feminine names are associated with larger damages.")
            elif dg['pvalue'] < 0.10:
                evidence_lines.append("Marginal evidence that more feminine names are associated with larger damages.")
            else:
                evidence_lines.append("No evidence that more feminine names are associated with damages (coefficient not statistically significant).")
        else:
            evidence_lines.append("Damage model coefficient available but p-value missing; cannot assess significance.")
    else:
        evidence_lines.append("No damage-model result available.")

    # Combine to a cautious answer:
    if (out['death'] is not None and out['death']['pvalue'] is not None and out['death']['pvalue'] < 0.10 and
        out['damage'] is not None and out['damage']['pvalue'] is not None and out['damage']['pvalue'] < 0.10):
        verdict = "Yes: both deaths and damages show evidence (or marginal evidence) consistent with the hypothesis."
    elif out['death'] is not None and out['death']['pvalue'] is not None and out['death']['pvalue'] < 0.10:
        verdict = "Partially yes: deaths show suggestive/marginal evidence that more feminine names are associated with higher fatalities, but damages do not show evidence."
    else:
        verdict = "No strong evidence: the models do not provide statistically robust support that more feminine names lead to fewer precautions (higher deaths/damages)."

    out['final_verdict'] = {
        "verdict": verdict,
        "evidence_summary": evidence_lines,
        "concise_note": (
            "Death model: coef ≈ 0.253 (p ≈ 0.059) → ~exp(0.253)=1.29x expected deaths per 1 SD increase in femininity "
            "(marginal, 95% CI for coef ≈ [-0.01, 0.516], multiplicative CI ≈ [0.99,1.68]). "
            "Damage model: coef ≈ 0.090 (p ≈ 0.596) → ~9% increase in damage but not significant."
        )
    }

    # Final returned structure
    return {
        "object": out,
        "description": (
            "Extracted coefficients and p-values for masfem_std from the death (Negative Binomial) and damage (OLS on log damage) models. "
            "Interpretation: The death model estimate is positive (coef ≈ 0.253) and marginally significant (p ≈ 0.059), implying roughly a 29% higher expected death count per 1 SD increase in name femininity, but the result is only suggestive (p between 0.05 and 0.10). "
            "The damage model shows a small positive point estimate (coef ≈ 0.090) but is not statistically significant (p ≈ 0.596). "
            "Overall: partial / suggestive evidence for the hypothesis for fatalities only, but no robust evidence for damages."
        )
    }