def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    interpretable effect sizes for the femininity predictors from the provided
    model_output dictionary.

    Returns a dict with:
      - "object": a nested dict with extracted statistics for each predictor
                  ('MasFem_z' and 'IsFemale') from the primary (negative
                  binomial / GLM) model and from the OLS robustness model.
      - "description": human-readable interpretation of whether the results
                       support the hypothesis that more-feminine names lead
                       to fewer precautions (i.e., more fatalities).

    The function is defensive: it falls back if some expected pieces are
    missing and reports what it could extract.
    """
    import numpy as np

    out = {"object": {}, "description": ""}

    # What predictors we care about
    predictors = ["MasFem_z", "IsFemale"]

    # Helper to safely extract values from various container types (Series, DataFrame, ndarray)
    def safe_get(container, names_list, pred, col=None):
        """
        container: Series, DataFrame, ndarray, or None
        names_list: list of parameter names in order for ndarray fallback
        pred: parameter name to extract
        col: for 2-column containers (like conf_int) choose column index 0 or 1, else None
        """
        if container is None:
            return None
        # Pandas-like with index (Series or single-column access)
        try:
            if hasattr(container, "loc") and pred in getattr(container, "index", []):
                if col is None:
                    return float(container.loc[pred])
                else:
                    return float(container.loc[pred, col])
            if hasattr(container, "index") and pred in container.index:
                # Series
                if col is None:
                    return float(container[pred])
                else:
                    # If it's DataFrame-like but without .loc, try iloc using index position
                    idx = list(container.index).index(pred)
                    return float(container.iloc[idx, col])
        except Exception:
            # fall through to ndarray handling
            pass

        # ndarray handling: need names_list to map pred -> row index
        if isinstance(container, np.ndarray) and names_list is not None:
            try:
                idx = names_list.index(pred)
            except ValueError:
                return None
            try:
                if col is None:
                    # 1D array or 2D with single column
                    if container.ndim == 1:
                        return float(container[idx])
                    elif container.ndim >= 2 and container.shape[1] == 1:
                        return float(container[idx, 0])
                    else:
                        # ambiguous, return the first element of the row
                        return float(container[idx, 0])
                else:
                    return float(container[idx, col])
            except Exception:
                return None

        return None

    def extract_from_results(res, model_type="GLM"):
        stats = {}
        if res is None:
            return stats

        params = getattr(res, "params", None)
        pvalues = getattr(res, "pvalues", None)
        bse = getattr(res, "bse", None)
        # conf_int may raise or return ndarray / DataFrame
        try:
            ci = res.conf_int()
        except Exception:
            ci = None

        # Build name list for ndarray fallbacks
        param_names = None
        # Prefer explicit names from params if available
        try:
            if params is not None and hasattr(params, "index"):
                param_names = list(params.index)
            else:
                # try various common attributes on results
                if hasattr(res, "model") and hasattr(res.model, "exog_names"):
                    param_names = list(res.model.exog_names)
                elif hasattr(res, "param_names"):
                    param_names = list(getattr(res, "param_names"))
                elif hasattr(res, "names"):
                    param_names = list(getattr(res, "names"))
                elif pvalues is not None and hasattr(pvalues, "index"):
                    param_names = list(pvalues.index)
                else:
                    # If params is ndarray but has length, and predictors are subset, create placeholder names
                    if isinstance(params, np.ndarray):
                        # we cannot reliably map, so leave as None
                        param_names = None
        except Exception:
            param_names = None

        for pred in predictors:
            # default None for each predictor
            stats[pred] = None

            # determine coefficient
            coef = safe_get(params, param_names, pred)
            if coef is None:
                # no coef found; leave None
                continue
            # get se, pval, ci
            se = safe_get(bse, param_names, pred)
            pval = safe_get(pvalues, param_names, pred)
            ci_low = safe_get(ci, param_names, pred, col=0)
            ci_high = safe_get(ci, param_names, pred, col=1)

            coef = float(coef)
            se = float(se) if se is not None else None
            pval = float(pval) if pval is not None else None

            if model_type.upper() in ("GLM", "NB", "POISSON", "NEGATIVEBINOMIAL"):
                # For count models, exponentiated coef = multiplicative change in expected count
                irr = float(np.exp(coef))
                irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
                interpretation = {
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                    "conf_low": ci_low,
                    "conf_high": ci_high,
                    "incidence_rate_ratio": irr,
                    "irr_conf_low": irr_ci_low,
                    "irr_conf_high": irr_ci_high,
                    "interpretation": (
                        "Multiplicative effect on expected deaths per one-unit increase "
                        "(e.g., for MasFem_z) or for IsFemale=1 vs 0."
                    )
                }
            else:
                # For OLS on log(1 + deaths), coef is change in log(1+deaths).
                # exp(coef)-1 approximates proportionate change in (1+deaths).
                prop_change = float(np.exp(coef) - 1.0)
                prop_ci_low = float(np.exp(ci_low) - 1.0) if ci_low is not None else None
                prop_ci_high = float(np.exp(ci_high) - 1.0) if ci_high is not None else None
                interpretation = {
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                    "conf_low": ci_low,
                    "conf_high": ci_high,
                    "proportional_change_in_1_plus_deaths": prop_change,
                    "prop_change_conf_low": prop_ci_low,
                    "prop_change_conf_high": prop_ci_high,
                    "interpretation": (
                        "Change in log(1 + deaths) per unit increase. exp(coef)-1 is the "
                        "proportional change in (1 + deaths)."
                    )
                }

            stats[pred] = interpretation
        return stats

    # Primary model: prefer 'nb_model' if present
    nb_res = model_output.get("nb_model", None)
    ols_res = model_output.get("ols_log_deaths", None)

    # Extract from NB/GLM (primary)
    nb_stats = extract_from_results(nb_res, model_type="GLM")
    ols_stats = extract_from_results(ols_res, model_type="OLS")

    out["object"]["nb_model"] = nb_stats
    out["object"]["ols_log_deaths"] = ols_stats

    # Derive a short conclusion focused on the hypothesis:
    # Hypothesis: more feminine names -> perceived less threatening -> fewer precautions -> MORE fatalities.
    def interpret_direction(stat_entry):
        if stat_entry is None:
            return "no_estimate"
        coef = stat_entry.get("coef", None)
        p = stat_entry.get("pvalue", None)
        if coef is None:
            return "no_estimate"
        if p is not None and p < 0.05:
            sig = "significant"
        else:
            sig = "not_significant"
        if coef > 0:
            direction = "positive"
        elif coef < 0:
            direction = "negative"
        else:
            direction = "zero"
        return f"{sig}_{direction}"

    # Interpret for MasFem_z primarily using NB model
    masfem_nb_interp = interpret_direction(nb_stats.get("MasFem_z") if nb_stats else None)
    isfemale_nb_interp = interpret_direction(nb_stats.get("IsFemale") if nb_stats else None)

    def fmt(value, fmt_spec="{:.4f}", na_str="NA"):
        try:
            if value is None:
                return na_str
            return fmt_spec.format(value)
        except Exception:
            return na_str

    summary_lines = []
    # Summarize MasFem_z
    if nb_stats.get("MasFem_z") is None:
        summary_lines.append("Primary model (negative binomial) did not return estimates for MasFem_z.")
    else:
        s = nb_stats["MasFem_z"]
        irr = s.get("incidence_rate_ratio", None)
        p = s.get("pvalue", None)
        irr_low = s.get("irr_conf_low", None)
        irr_high = s.get("irr_conf_high", None)
        summary_lines.append(
            f"Negative binomial: MasFem_z coef = {fmt(s.get('coef'))}, IRR = {fmt(irr)} "
            f"(95% CI IRR [{fmt(irr_low)}, {fmt(irr_high)}] if available), p = {fmt(p, '{:.3g}')}. "
        )

    # Summarize IsFemale
    if nb_stats.get("IsFemale") is None:
        summary_lines.append("Primary model did not return estimates for IsFemale.")
    else:
        s = nb_stats["IsFemale"]
        irr = s.get("incidence_rate_ratio", None)
        p = s.get("pvalue", None)
        irr_low = s.get("irr_conf_low", None)
        irr_high = s.get("irr_conf_high", None)
        summary_lines.append(
            f"Negative binomial: IsFemale coef = {fmt(s.get('coef'))}, IRR = {fmt(irr)} "
            f"(95% CI IRR [{fmt(irr_low)}, {fmt(irr_high)}] if available), p = {fmt(p, '{:.3g}')}. "
        )

    # Add robustness (OLS) summary
    if ols_stats:
        if ols_stats.get("MasFem_z") is not None:
            s = ols_stats["MasFem_z"]
            prop = s.get("proportional_change_in_1_plus_deaths", None)
            p = s.get("pvalue", None)
            summary_lines.append(
                f"OLS on log(1+deaths) (robust SE): MasFem_z coef = {fmt(s.get('coef'))}, "
                f"implying approx. {fmt((prop * 100) if prop is not None else None, '{:.2f}')}% change in (1+deaths) per unit, p = {fmt(p, '{:.3g}')}. "
            )
        if ols_stats.get("IsFemale") is not None:
            s = ols_stats["IsFemale"]
            prop = s.get("proportional_change_in_1_plus_deaths", None)
            p = s.get("pvalue", None)
            summary_lines.append(
                f"OLS on log(1+deaths) (robust SE): IsFemale coef = {fmt(s.get('coef'))}, "
                f"implying approx. {fmt((prop * 100) if prop is not None else None, '{:.2f}')}% change in (1+deaths) for female vs male, p = {fmt(p, '{:.3g}')}. "
            )

    # Formal conclusion sentence: check whether MasFem_z shows a statistically significant positive association with Deaths
    conclusion = ""
    if masfem_nb_interp == "significant_positive":
        conclusion = (
            "Conclusion: The primary model shows a statistically significant POSITIVE association between "
            "MasFem_z (more feminine) and fatalities — i.e., more feminine names are associated with MORE deaths, "
            "which is consistent with the hypothesis that feminine names lead to fewer precautions."
        )
    elif masfem_nb_interp == "significant_negative":
        conclusion = (
            "Conclusion: The primary model shows a statistically significant NEGATIVE association between "
            "MasFem_z and fatalities — i.e., more feminine names are associated with FEWER deaths, "
            "which is contrary to the hypothesis."
        )
    elif masfem_nb_interp in ("not_significant_positive", "not_significant_negative", "not_significant_zero"):
        direction = masfem_nb_interp.split("_", 1)[1]
        conclusion = (
            f"Conclusion: The primary model estimates a {direction} association between MasFem_z and fatalities "
            "but it is not statistically significant (p >= 0.05). This does not provide strong evidence for the hypothesis."
        )
    else:
        conclusion = "Conclusion: Could not determine a clear result for MasFem_z from the primary model."

    # Also mention IsFemale briefly
    if isfemale_nb_interp == "significant_positive":
        conclusion += " The binary IsFemale predictor (female name) is also significantly associated with MORE deaths."
    elif isfemale_nb_interp == "significant_negative":
        conclusion += " The binary IsFemale predictor is significantly associated with FEWER deaths."
    elif isfemale_nb_interp.startswith("not_significant"):
        conclusion += " The binary IsFemale predictor is not statistically significant in the primary model."

    out["description"] = "\n".join(summary_lines + ["", conclusion])

    return out