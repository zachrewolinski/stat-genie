def extract_final_answer(model_output):
    """
    Extracts key statistics related to the femininity effect on hurricane deaths
    from the models returned by the modeling function.

    Returns:
      {
        "object": {
           "nb_masfem": { ... stats for masfem_z ... },
           "nb_gender": { ... stats for gender_female ... },
           "ols_log_alldeaths": { ... stats for masfem_z ... },
           "ols_log_ndam15": { ... stats for masfem_z ... (if present) ... },
           "conclusion": "<Yes/No/Unclear> - short rationale"
        },
        "description": "Human readable summary and interpretation"
      }
    """
    import numpy as np

    out = {}
    desc_lines = []

    # Helper to safely extract parameter stats from a statsmodels result
    def extract_param_stats(model, param):
        try:
            params = model.params
            bse = model.bse
            pvalues = model.pvalues
            ci = model.conf_int()
            coef = float(params[param])
            se = float(bse[param])
            p = float(pvalues[param])
            ci_low, ci_high = float(ci.loc[param, 0]), float(ci.loc[param, 1])
            return {
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "ci_95": [ci_low, ci_high]
            }
        except Exception as e:
            return {"error": f"Could not extract '{param}': {e}"}

    # 1) Primary model: Negative Binomial with masfem_z
    if "nb_masfem" in model_output and hasattr(model_output["nb_masfem"], "params"):
        m = model_output["nb_masfem"]
        stats_m = extract_param_stats(m, "masfem_z")
        if "error" not in stats_m:
            # compute IRR and CI on IRR scale
            irr = float(np.exp(stats_m["coef"]))
            irr_ci = [float(np.exp(stats_m["ci_95"][0])), float(np.exp(stats_m["ci_95"][1]))]
            stats_m.update({"incidence_rate_ratio": irr, "irr_95_ci": irr_ci})
            out["nb_masfem"] = stats_m
            desc_lines.append(
                "Primary NB model (masfem_z): coef = {coef:.4f}, SE = {std_err:.4f}, p = {p_value:.4g}; "
                "IRR = {irr:.4f} (95% CI [{ci0:.4f}, {ci1:.4f}])."
                .format(coef=stats_m["coef"], std_err=stats_m["std_err"],
                        p_value=stats_m["p_value"], irr=stats_m["incidence_rate_ratio"],
                        ci0=stats_m["irr_95_ci"][0], ci1=stats_m["irr_95_ci"][1])
            )
        else:
            out["nb_masfem"] = stats_m
            desc_lines.append(f"Primary NB model (masfem_z): {stats_m['error']}")
    else:
        out["nb_masfem"] = {"error": "nb_masfem not present or not a fitted model"}
        desc_lines.append("Primary NB model (masfem_z) not available.")

    # 2) Robustness: NB with binary gender indicator
    if "nb_gender" in model_output and hasattr(model_output["nb_gender"], "params"):
        mg = model_output["nb_gender"]
        stats_g = extract_param_stats(mg, "gender_female")
        if "error" not in stats_g:
            irr_g = float(np.exp(stats_g["coef"]))
            irr_g_ci = [float(np.exp(stats_g["ci_95"][0])), float(np.exp(stats_g["ci_95"][1]))]
            stats_g.update({"incidence_rate_ratio": irr_g, "irr_95_ci": irr_g_ci})
            out["nb_gender"] = stats_g
            desc_lines.append(
                "NB gender robustness: coef (female) = {coef:.4f}, SE = {std_err:.4f}, p = {p_value:.4g}; "
                "IRR = {irr:.4f} (95% CI [{ci0:.4f}, {ci1:.4f}])."
                .format(coef=stats_g["coef"], std_err=stats_g["std_err"],
                        p_value=stats_g["p_value"], irr=stats_g["incidence_rate_ratio"],
                        ci0=stats_g["irr_95_ci"][0], ci1=stats_g["irr_95_ci"][1])
            )
        else:
            out["nb_gender"] = stats_g
            desc_lines.append(f"NB gender robustness: {stats_g['error']}")
    else:
        out["nb_gender"] = {"error": "nb_gender not present or not a fitted model"}
        desc_lines.append("NB gender robustness model not available.")

    # 3) Sensitivity: OLS on log(1 + alldeaths)
    if "ols_log_alldeaths" in model_output and hasattr(model_output["ols_log_alldeaths"], "params"):
        mo = model_output["ols_log_alldeaths"]
        stats_o = extract_param_stats(mo, "masfem_z")
        if "error" not in stats_o:
            # For OLS on log outcome, exponentiating is not directly an IRR but the sign/percent change approx:
            approx_pct_change = (np.exp(stats_o["coef"]) - 1) * 100.0  # approximate percent change in level after retransformation
            stats_o.update({"approx_percent_change_in_1plus_deaths": float(approx_pct_change)})
            out["ols_log_alldeaths"] = stats_o
            desc_lines.append(
                "OLS log(1+deaths) sensitivity: coef = {coef:.4f}, SE = {std_err:.4f}, p = {p_value:.4g}; "
                "approx % change (exp(coef)-1) = {pct:.2f}%."
                .format(coef=stats_o["coef"], std_err=stats_o["std_err"],
                        p_value=stats_o["p_value"], pct=stats_o["approx_percent_change_in_1plus_deaths"])
            )
        else:
            out["ols_log_alldeaths"] = stats_o
            desc_lines.append(f"OLS log sensitivity: {stats_o['error']}")
    else:
        out["ols_log_alldeaths"] = {"error": "ols_log_alldeaths not present or not a fitted model"}
        desc_lines.append("OLS log(1+alldeaths) model not available.")

    # 4) Optional: OLS on log(1 + ndam15) economic damage
    if "ols_log_ndam15" in model_output and hasattr(model_output["ols_log_ndam15"], "params"):
        md = model_output["ols_log_ndam15"]
        stats_d = extract_param_stats(md, "masfem_z")
        if "error" not in stats_d:
            approx_pct_change_d = (np.exp(stats_d["coef"]) - 1) * 100.0
            stats_d.update({"approx_percent_change_in_damage": float(approx_pct_change_d)})
            out["ols_log_ndam15"] = stats_d
            desc_lines.append(
                "OLS log(1+damage) sensitivity: coef = {coef:.4f}, SE = {std_err:.4f}, p = {p_value:.4g}; "
                "approx % change = {pct:.2f}%."
                .format(coef=stats_d["coef"], std_err=stats_d["std_err"],
                        p_value=stats_d["p_value"], pct=stats_d["approx_percent_change_in_damage"])
            )
        else:
            out["ols_log_ndam15"] = stats_d
            desc_lines.append(f"OLS damage sensitivity: {stats_d['error']}")
    else:
        out["ols_log_ndam15"] = {"error": "ols_log_ndam15 not present or not a fitted model"}
        desc_lines.append("OLS log(1+ndam15) model not available.")

    # Determine a simple rule for supporting the hypothesis:
    # Hypothesis: more feminine names -> higher deaths (positive association).
    # We consider the primary NB model decisive: require coef>0 and p<0.05 to say "Yes".
    support_primary = False
    try:
        sp = out.get("nb_masfem", {})
        if "coef" in sp and sp["coef"] > 0 and sp["p_value"] < 0.05:
            support_primary = True
    except Exception:
        support_primary = False

    # Check robustness consistency (binary gender or OLS)
    robust_evidence = False
    try:
        # treat any positive significant effect in robustness checks as supporting
        for key in ("nb_gender", "ols_log_alldeaths"):
            s = out.get(key, {})
            if "coef" in s and s["coef"] > 0 and s["p_value"] < 0.05:
                robust_evidence = True
    except Exception:
        robust_evidence = False

    if support_primary:
        if robust_evidence:
            conclusion = "Yes — primary NB model shows a positive, statistically significant association (p<0.05) and at least one robustness check is consistent."
        else:
            conclusion = "Yes (primary) — primary NB model shows a positive, statistically significant association (p<0.05), but robustness checks are not consistently significant."
    else:
        # If primary not significant but robustness checks show consistent positive significant effects, say "Unclear/Weak"
        if robust_evidence:
            conclusion = "Unclear/Weak — primary NB model does not show a significant positive effect, but at least one robustness check shows a positive significant effect."
        else:
            conclusion = "No — there is no consistent statistically significant evidence that more-feminine names increase deaths in these models."

    out["conclusion"] = conclusion

    # Compose human-readable description
    description = (
        "Extracted statistics for the effect of name femininity on hurricane deaths and robustness checks.\n"
        + "\n".join(desc_lines)
        + f"\n\nConclusion rule: we treat the primary NB (masfem_z) as the main test. Result: {conclusion}"
        + "\n\nInterpretation: A positive coefficient means that more-feminine names are associated with higher fatalities (consistent with the hypothesis that feminine names reduce perceived threat and precautions). "
        "The IRR (exp(coef)) from the NB model gives the multiplicative change in expected death counts per 1 SD increase in femininity. "
        "The OLS coefficients are on log(1+outcome) and the reported approximate % change is (exp(coef)-1)*100."
    )

    return {"object": out, "description": description}