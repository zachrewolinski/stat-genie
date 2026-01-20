def extract_final_answer(model_output):
    """
    Extract relevant statistics for 'NameFemininity' (continuous) and 'IsFemaleName' (binary)
    from the provided model_output dict.

    Expects model_output to contain:
      - 'nb_model' : fitted statsmodels Negative Binomial model results (GLM or discrete)
      - 'ols_log_model' : fitted statsmodels OLS results on log1p(Deaths)
      - optionally 'predictors' : list of predictor names

    Returns a dict with keys:
      - "object": nested dict with extracted numeric statistics for each predictor and model
      - "description": brief interpretation of the key results in the context of the hypothesis
    """
    import numpy as np

    # Helpers to safely extract attributes
    def safe_attr(obj, name, default=None):
        return getattr(obj, name, default)

    def get_param_table(result, var):
        """
        Return dict with coef, se, pval, ci_low, ci_upp for var from a statsmodels result.
        """
        out = {}
        params = safe_attr(result, "params", None)
        bse = safe_attr(result, "bse", None)
        pvalues = safe_attr(result, "pvalues", None)
        try:
            ci = result.conf_int()
        except Exception:
            # some older objects may have conf_int as a function - try calling
            try:
                ci = result.conf_int()
            except Exception:
                ci = None

        if params is None or var not in params:
            return None

        coef = float(params[var])
        se = float(bse[var]) if (bse is not None and var in bse) else None
        pval = float(pvalues[var]) if (pvalues is not None and var in pvalues) else None

        if ci is not None:
            # ci may be DataFrame or ndarray; indexable by var
            try:
                lower, upper = ci.loc[var][0], ci.loc[var][1]
            except Exception:
                try:
                    # if ci is ndarray and we know order, fallback to None
                    lower, upper = float(ci[params.index.get_loc(var), 0]), float(ci[params.index.get_loc(var), 1])
                except Exception:
                    # last resort: try to index by position if var exists in params index
                    lower, upper = None, None
        else:
            lower, upper = None, None

        out.update({
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": float(lower) if lower is not None else None,
            "ci_upper": float(upper) if upper is not None else None,
        })
        return out

    # Validate input
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict as returned by the modeling function.")

    nb = model_output.get("nb_model", None)
    ols = model_output.get("ols_log_model", None)
    if nb is None and ols is None:
        raise KeyError("model_output must contain at least 'nb_model' or 'ols_log_model'.")

    predictors = model_output.get("predictors", ['NameFemininity', 'IsFemaleName'])
    targets = [p for p in ['NameFemininity', 'IsFemaleName'] if p in predictors]

    results_obj = {"negative_binomial": {}, "ols_log1p": {}}

    # Extract for NB model (incidence rate ratios)
    if nb is not None:
        for var in targets:
            tbl = get_param_table(nb, var)
            if tbl is None:
                results_obj["negative_binomial"][var] = None
                continue
            # compute IRR and CI on IRR scale
            irr = np.exp(tbl["coef"])
            irr_ci_lower = np.exp(tbl["ci_lower"]) if tbl["ci_lower"] is not None else None
            irr_ci_upper = np.exp(tbl["ci_upper"]) if tbl["ci_upper"] is not None else None
            pct_change = (irr - 1.0) * 100.0  # percent change in expected count per unit increase
            pct_ci_lower = (irr_ci_lower - 1.0) * 100.0 if irr_ci_lower is not None else None
            pct_ci_upper = (irr_ci_upper - 1.0) * 100.0 if irr_ci_upper is not None else None

            results_obj["negative_binomial"][var] = {
                "coef_log_count": tbl["coef"],
                "se": tbl["se"],
                "pvalue": tbl["pvalue"],
                "ci_log_count": [tbl["ci_lower"], tbl["ci_upper"]],
                "IRR": float(irr),
                "IRR_CI": [float(irr_ci_lower) if irr_ci_lower is not None else None,
                           float(irr_ci_upper) if irr_ci_upper is not None else None],
                "approx_pct_change_in_deaths_per_unit": float(pct_change),
                "approx_pct_change_CI": [float(pct_ci_lower) if pct_ci_lower is not None else None,
                                         float(pct_ci_upper) if pct_ci_upper is not None else None]
            }

        # add some model diagnostics if available
        try:
            results_obj["negative_binomial"]["_nobs"] = int(nb.nobs)
        except Exception:
            pass
        try:
            results_obj["negative_binomial"]["_aic"] = float(nb.aic)
        except Exception:
            pass

    # Extract for OLS on log1p(Deaths)
    if ols is not None:
        for var in targets:
            tbl = get_param_table(ols, var)
            if tbl is None:
                results_obj["ols_log1p"][var] = None
                continue
            # For log1p(Deaths) model, exponentiating coef gives multiplicative effect on (Deaths+1) approximately
            approx_multiplier = np.exp(tbl["coef"])
            approx_pct = (approx_multiplier - 1.0) * 100.0
            ci_low = tbl["ci_lower"]
            ci_up = tbl["ci_upper"]
            approx_multiplier_ci_low = np.exp(ci_low) if ci_low is not None else None
            approx_multiplier_ci_up = np.exp(ci_up) if ci_up is not None else None
            approx_pct_ci_low = (approx_multiplier_ci_low - 1.0) * 100.0 if approx_multiplier_ci_low is not None else None
            approx_pct_ci_up = (approx_multiplier_ci_up - 1.0) * 100.0 if approx_multiplier_ci_up is not None else None

            results_obj["ols_log1p"][var] = {
                "coef_on_log1p_deaths": tbl["coef"],
                "se": tbl["se"],
                "pvalue": tbl["pvalue"],
                "ci_log1p": [tbl["ci_lower"], tbl["ci_upper"]],
                "approx_multiplier_on_deaths_plus1": float(approx_multiplier),
                "approx_pct_change_on_deaths_plus1_per_unit": float(approx_pct),
                "approx_pct_change_CI": [float(approx_pct_ci_low) if approx_pct_ci_low is not None else None,
                                         float(approx_pct_ci_up) if approx_pct_ci_up is not None else None]
            }

        try:
            results_obj["ols_log1p"]["_nobs"] = int(ols.nobs)
        except Exception:
            pass
        try:
            results_obj["ols_log1p"]["_rsquared"] = float(ols.rsquared)
        except Exception:
            pass

    # Brief interpretation of the key signs and significance for both predictors
    def interpret_section(section_dict, model_name):
        lines = []
        for var, stats in section_dict.items():
            if var.startswith("_"):
                continue
            if stats is None:
                lines.append(f"{model_name}: No results for {var}.")
                continue
            coef = stats.get("coef_log_count") if model_name == "negative_binomial" else stats.get("coef_on_log1p_deaths")
            p = stats.get("pvalue")
            direction = "positive" if coef > 0 else ("zero" if np.isclose(coef, 0.0) else "negative")
            sig = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
            # Determine substantive interpretation
            if model_name == "negative_binomial":
                irr = stats.get("IRR")
                pct = stats.get("approx_pct_change_in_deaths_per_unit")
                lines.append(f"{model_name} - {var}: coef={coef:.4g}, IRR={irr:.4g}, p={p:.4g} ({direction}, {sig}). Approx. {pct:.2f}% change in expected deaths per unit increase.")
            else:
                pct = stats.get("approx_pct_change_on_deaths_plus1_per_unit")
                lines.append(f"{model_name} - {var}: coef={coef:.4g}, p={p:.4g} ({direction}, {sig}). Approx. {pct:.2f}% change in (Deaths+1) per unit increase (multiplicative).")
        return " ".join(lines)

    nb_interp = interpret_section(results_obj["negative_binomial"], "negative_binomial") if results_obj["negative_binomial"] else ""
    ols_interp = interpret_section(results_obj["ols_log1p"], "ols_log1p") if results_obj["ols_log1p"] else ""

    description_lines = []
    if nb_interp:
        description_lines.append(nb_interp)
    if ols_interp:
        description_lines.append(ols_interp)
    description_lines.append(
        "Interpretation guidance: For the hypothesis that more feminine hurricane names lead to fewer precautions (and hence MORE fatalities), "
        "a positive and statistically significant coefficient on NameFemininity or IsFemaleName would support that claim. A negative coefficient would "
        "contradict it. The negative binomial IRR >1 indicates higher expected counts (deaths) per unit increase in the predictor; IRR <1 indicates fewer expected deaths."
    )

    return {
        "object": results_obj,
        "description": " ".join(description_lines)
    }