def extract_final_answer(model_output):
    """
    Extract key statistics for the main hypothesis from the provided statsmodels results.

    Expects model_output to be a dict with keys:
      - 'main_deaths_model' -> statsmodels RegressionResultsWrapper (contains 'FemScore')
      - 'binary_deaths_model' -> statsmodels RegressionResultsWrapper (contains 'FemaleBinary')
      - 'damage_model' -> statsmodels RegressionResultsWrapper (contains 'FemScore')

    Returns:
      {
        "object": {
           "main_deaths": { ... stats ... },
           "binary_deaths": { ... stats ... },
           "damage": { ... stats ... }
        },
        "description": "Plain-language summary of results and (direction/significance) relative to the hypothesis"
      }
    """
    def _get_stats(model, varname):
        out = {
            "variable": varname,
            "present": False,
            "coef": None,
            "se": None,
            "t": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "nobs": None
        }
        try:
            params = model.params
            if varname not in params.index:
                return out
            out["present"] = True
            out["coef"] = float(params.loc[varname])
            # standard error, t, p
            try:
                out["se"] = float(model.bse.loc[varname])
            except Exception:
                out["se"] = float(model.bse[varname])
            try:
                out["t"] = float(model.tvalues.loc[varname])
            except Exception:
                out["t"] = float(model.tvalues[varname])
            try:
                out["p_value"] = float(model.pvalues.loc[varname])
            except Exception:
                out["p_value"] = float(model.pvalues[varname])
            # confidence interval
            try:
                ci = model.conf_int().loc[varname]
                out["ci_lower"] = float(ci[0])
                out["ci_upper"] = float(ci[1])
            except Exception:
                ci_arr = model.conf_int().values
                # try to find row by matching varname in index
                try:
                    row = model.conf_int().loc[varname].values
                    out["ci_lower"], out["ci_upper"] = float(row[0]), float(row[1])
                except Exception:
                    # fallback: set None
                    out["ci_lower"], out["ci_upper"] = None, None
            # nobs
            try:
                out["nobs"] = int(model.nobs)
            except Exception:
                try:
                    out["nobs"] = int(getattr(model, "nobs"))
                except Exception:
                    out["nobs"] = None
        except Exception:
            # return what we have (mostly Nones)
            pass
        return out

    results = {}
    descriptions = []

    # Safely extract models from input dict
    main_model = model_output.get('main_deaths_model')
    bin_model = model_output.get('binary_deaths_model')
    dmg_model = model_output.get('damage_model')

    # Main model: FemScore -> LogDeaths
    main_stats = None
    if main_model is not None:
        main_stats = _get_stats(main_model, "FemScore")
        results["main_deaths"] = main_stats
        if not main_stats["present"]:
            descriptions.append("Main model: variable 'FemScore' not present in model object.")
        else:
            sig = main_stats["p_value"] < 0.05
            direction = "positive" if main_stats["coef"] > 0 else "negative" if main_stats["coef"] < 0 else "null"
            descriptions.append(
                f"Main model (LogDeaths ~ FemScore + controls): coef={main_stats['coef']:.4f}, SE={main_stats['se']:.4f}, "
                f"p={main_stats['p_value']:.4f}, 95% CI=[{main_stats['ci_lower']:.4f}, {main_stats['ci_upper']:.4f}], "
                f"n={main_stats['nobs']}. This is a {direction} effect; "
                + ("statistically significant (p<0.05)." if sig else "not statistically significant (p>=0.05).")
            )
    else:
        descriptions.append("Main model ('main_deaths_model') not found in model_output.")

    # Binary-name robustness: FemaleBinary -> LogDeaths
    bin_stats = None
    if bin_model is not None:
        bin_stats = _get_stats(bin_model, "FemaleBinary")
        results["binary_deaths"] = bin_stats
        if not bin_stats["present"]:
            descriptions.append("Binary robustness model: variable 'FemaleBinary' not present in model object.")
        else:
            sig = bin_stats["p_value"] < 0.05
            direction = "positive" if bin_stats["coef"] > 0 else "negative" if bin_stats["coef"] < 0 else "null"
            descriptions.append(
                f"Binary robustness (LogDeaths ~ FemaleBinary + controls): coef={bin_stats['coef']:.4f}, SE={bin_stats['se']:.4f}, "
                f"p={bin_stats['p_value']:.4f}, 95% CI=[{bin_stats['ci_lower']:.4f}, {bin_stats['ci_upper']:.4f}], "
                f"n={bin_stats['nobs']}. This is a {direction} effect; "
                + ("statistically significant (p<0.05)." if sig else "not statistically significant (p>=0.05).")
            )
    else:
        descriptions.append("Binary robustness model ('binary_deaths_model') not found in model_output.")

    # Damage robustness: FemScore -> LogDamage2015
    dmg_stats = None
    if dmg_model is not None:
        dmg_stats = _get_stats(dmg_model, "FemScore")
        results["damage"] = dmg_stats
        if not dmg_stats["present"]:
            descriptions.append("Damage robustness model: variable 'FemScore' not present in model object.")
        else:
            sig = dmg_stats["p_value"] < 0.05
            direction = "positive" if dmg_stats["coef"] > 0 else "negative" if dmg_stats["coef"] < 0 else "null"
            descriptions.append(
                f"Damage robustness (LogDamage2015 ~ FemScore + controls): coef={dmg_stats['coef']:.4f}, SE={dmg_stats['se']:.4f}, "
                f"p={dmg_stats['p_value']:.4f}, 95% CI=[{dmg_stats['ci_lower']:.4f}, {dmg_stats['ci_upper']:.4f}], "
                f"n={dmg_stats['nobs']}. This is a {direction} effect; "
                + ("statistically significant (p<0.05)." if sig else "not statistically significant (p>=0.05).")
            )
    else:
        descriptions.append("Damage robustness model ('damage_model') not found in model_output.")

    # Summarize relevance to hypothesis
    # Hypothesis: more feminine names -> higher fatalities (positive coef on FemScore or FemaleBinary)
    def _interpret_section(stat):
        if stat is None or not stat.get("present"):
            return None
        coef = stat["coef"]
        p = stat["p_value"]
        if p is None:
            return None
        if p < 0.05:
            sig_text = "statistically significant"
        elif p < 0.1:
            sig_text = "marginally significant (p<0.10)"
        else:
            sig_text = "not statistically significant"
        direction = "consistent with" if coef > 0 else "in the opposite direction to"
        return f"The effect is {sig_text} and the sign is {direction} the hypothesis (coef={coef:.4f})."

    interp_main = _interpret_section(main_stats)
    interp_bin = _interpret_section(bin_stats)
    interp_dmg = _interpret_section(dmg_stats)

    summary_lines = []
    if interp_main:
        summary_lines.append("Main model: " + interp_main)
    if interp_bin:
        summary_lines.append("Binary robustness: " + interp_bin)
    if interp_dmg:
        summary_lines.append("Damage robustness: " + interp_dmg)
    if not summary_lines:
        summary_lines.append("No interpretable statistics were extracted to evaluate the hypothesis.")

    full_description = "\n".join(descriptions) + "\n\nSummary interpretation:\n" + "\n".join(summary_lines)

    return {"object": results, "description": full_description}