def extract_final_answer(model_output):
    """
    Extracts key statistics for the independent variable 'masfem_c' from the provided
    model_output dictionary containing:
      - 'ols_log_damage': OLS RegressionResultsWrapper (HC3 robust SEs)
      - 'glm_nb_deaths': GLMResultsWrapper (Negative Binomial or Poisson)
    
    Returns a dictionary with:
      - "object": dict containing numeric results for ols and glm and a conclusion on the hypothesis
      - "description": human-readable explanation of the extracted statistics and interpretation
    """
    import numpy as np

    out = {
        "object": None,
        "description": None
    }

    # Helper to safely extract stats for a given fitted results object
    def extract_from_results(res, varname):
        info = {}
        if res is None:
            return None
        # Some results objects expose params as pandas Series
        try:
            coef = float(res.params[varname])
        except Exception:
            raise KeyError(f"Variable '{varname}' not found in model params.")
        # p-value, se, conf int (these should be present for statsmodels results)
        se = float(res.bse[varname]) if hasattr(res, "bse") else None
        pval = float(res.pvalues[varname]) if hasattr(res, "pvalues") else None
        # conf_int may be method or attribute
        try:
            ci = res.conf_int().loc[varname].astype(float)
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            # try alternative indexing
            ci_arr = np.asarray(res.conf_int())
            # Try to find row corresponding to varname if params has index
            try:
                idx = list(res.params.index).index(varname)
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                ci_low, ci_high = None, None

        info.update({
            "coef": coef,
            "se": se,
            "p_value": pval,
            "ci_95": [ci_low, ci_high]
        })
        return info

    try:
        ols_res = model_output.get('ols_log_damage')
        glm_res = model_output.get('glm_nb_deaths')

        # Extract OLS statistics
        ols_stats = extract_from_results(ols_res, 'masfem_c')

        # Convert OLS log-damage coefficient to percent change in damage:
        # percent change = (exp(coef) - 1) * 100
        if ols_stats is not None:
            coef = ols_stats["coef"]
            ci_low, ci_high = ols_stats["ci_95"]
            pct_change = (np.exp(coef) - 1.0) * 100.0
            pct_ci_low = (np.exp(ci_low) - 1.0) * 100.0 if ci_low is not None else None
            pct_ci_high = (np.exp(ci_high) - 1.0) * 100.0 if ci_high is not None else None
            ols_stats.update({
                "percent_change_in_damage_per_unit_masfem_c": pct_change,
                "percent_change_95ci": [pct_ci_low, pct_ci_high]
            })

        # Extract GLM (deaths) statistics
        glm_stats = extract_from_results(glm_res, 'masfem_c')
        # For count model, exponentiate to get incidence rate ratio (IRR)
        if glm_stats is not None:
            coef_g = glm_stats["coef"]
            ci_low_g, ci_high_g = glm_stats["ci_95"]
            irr = np.exp(coef_g)
            irr_ci_low = np.exp(ci_low_g) if ci_low_g is not None else None
            irr_ci_high = np.exp(ci_high_g) if ci_high_g is not None else None
            glm_stats.update({
                "incidence_rate_ratio": irr,
                "irr_95ci": [irr_ci_low, irr_ci_high]
            })

        # Form a simple hypothesis conclusion using OLS as primary test:
        conclusion = {}
        if ols_stats is None:
            conclusion_text = "OLS results for 'masfem_c' not available; cannot evaluate hypothesis."
            supports = None
        else:
            coef = ols_stats["coef"]
            p = ols_stats["p_value"]
            # Convention: two-sided p < 0.05 considered 'statistically significant'
            significant = (p is not None) and (p < 0.05)
            # Hypothesis expects more feminine names -> less damage (negative effect)
            if significant and coef < 0:
                supports = True
                conclusion_text = (
                    "Result supports the hypothesis: coefficient on 'masfem_c' is negative "
                    f"(coef={coef:.4f}, p={p:.3f}), corresponding to a "
                    f"{ols_stats['percent_change_in_damage_per_unit_masfem_c']:.2f}% "
                    "decrease in expected damage per one-unit increase in femininity (95% CI "
                    f"[{ols_stats['percent_change_95ci'][0]:.2f}%, {ols_stats['percent_change_95ci'][1]:.2f}%])."
                )
            elif significant and coef > 0:
                supports = False
                conclusion_text = (
                    "Result contradicts the hypothesis: coefficient on 'masfem_c' is positive "
                    f"(coef={coef:.4f}, p={p:.3f}), corresponding to a "
                    f"{ols_stats['percent_change_in_damage_per_unit_masfem_c']:.2f}% "
                    "increase in expected damage per one-unit increase in femininity."
                )
            else:
                supports = False
                # Non-significant => do not claim support
                conclusion_text = (
                    "No statistically significant evidence to support the hypothesis: coefficient on 'masfem_c' "
                    f"is {coef:.4f} with p={p:.3f}. The point estimate corresponds to a "
                    f"{ols_stats['percent_change_in_damage_per_unit_masfem_c']:.2f}% change in damage per unit, "
                    "but the 95% confidence interval includes zero (no effect)."
                )

            conclusion.update({
                "supports_hypothesis": supports,
                "reason": conclusion_text,
                "ols_significant": significant,
                "ols_coef": coef,
                "ols_p_value": p
            })

        # Build final object
        final_obj = {
            "ols_log_damage": ols_stats,
            "glm_nb_deaths": glm_stats,
            "conclusion": conclusion
        }

        # Build a descriptive summary
        desc_lines = []
        desc_lines.append("Extracted estimates for predictor 'masfem_c' (femininity of hurricane name).")
        if ols_stats is not None:
            desc_lines.append(
                ("OLS on log-damage: coef = {coef:.4f}, se = {se:.4f}, p = {p:.3f}, "
                 "95% CI = [{cil:.4f}, {cih:.4f}]. Interpreted as {pct:.2f}% change in expected damage "
                 "per one-unit increase in femininity (95% CI [{pctl:.2f}%, {pcth:.2f}%]).")
                .format(coef=ols_stats["coef"], se=ols_stats["se"], p=ols_stats["p_value"],
                        cil=ols_stats["ci_95"][0], cih=ols_stats["ci_95"][1],
                        pct=ols_stats["percent_change_in_damage_per_unit_masfem_c"],
                        pctl=ols_stats["percent_change_95ci"][0], pcth=ols_stats["percent_change_95ci"][1])
            )
        else:
            desc_lines.append("OLS results not available or 'masfem_c' not found in OLS model.")

        if glm_stats is not None:
            desc_lines.append(
                ("Count model on deaths: coef = {coef:.4f}, p = {p:.3f}, IRR = {irr:.4f}, "
                 "IRR 95% CI = [{irl:.4f}, {irh:.4f}]. Interpreted as multiplicative change in expected deaths "
                 "per one-unit increase in femininity.")
                .format(coef=glm_stats["coef"], p=glm_stats["p_value"],
                        irr=glm_stats["incidence_rate_ratio"],
                        irl=glm_stats["irr_95ci"][0], irh=glm_stats["irr_95ci"][1])
            )
        else:
            desc_lines.append("Count model results not available or 'masfem_c' not found in GLM model.")

        # Add final decision text
        if conclusion:
            desc_lines.append("Conclusion: " + conclusion["reason"])

        out["object"] = final_obj
        out["description"] = " ".join(desc_lines)

    except KeyError as e:
        out["object"] = None
        out["description"] = f"Failed to extract variable statistics: {str(e)}"
    except Exception as e:
        out["object"] = None
        out["description"] = f"An unexpected error occurred during extraction: {type(e).__name__}: {e}"

    return out