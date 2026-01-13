def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of hurricane name femininity on fatalities
    from the provided model_output dict.

    Expects model_output to be a dict with keys:
      - 'ols_logfatalities_masfem' : statsmodels OLS results (LogFatalities ~ MasFem_z + ...)
      - 'ols_logfatalities_femalebin' : statsmodels OLS results (LogFatalities ~ Female + ...)
      - 'count_model_fatalities_nb_or_poisson' : statsmodels GLM (NB or Poisson) results on raw Fatalities

    Returns a dict:
      - "object": dict with extracted numeric summaries for each model (coef, se, p, CI, effect interpretation)
      - "description": short natural-language interpretation about whether evidence supports the hypothesis
    """
    import math
    import numpy as np

    def safe_get_stats(res, var):
        """Attempt to extract coefficient, se, pvalue, conf_int for variable var from a statsmodels result."""
        if res is None:
            return None
        try:
            params = getattr(res, "params", None)
            if params is None:
                return None
            coef = float(params[var])
        except Exception:
            # var not found
            return None

        # standard error
        se = None
        try:
            bse = getattr(res, "bse", None)
            if bse is not None:
                se = float(bse[var])
        except Exception:
            se = None

        # p-value
        pval = None
        try:
            pvals = getattr(res, "pvalues", None)
            if pvals is not None:
                pval = float(pvals[var])
        except Exception:
            pval = None

        # conf int
        ci_lower, ci_upper = (None, None)
        try:
            ci = res.conf_int()
            # conf_int may return a DataFrame or ndarray; handle both
            if hasattr(ci, "loc"):
                ci_lower, ci_upper = float(ci.loc[var].iloc[0]), float(ci.loc[var].iloc[1])
            else:
                # assume rows correspond to params in order
                idx = list(res.params.index).index(var)
                ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
        except Exception:
            ci_lower, ci_upper = (None, None)

        # test stat (t or z)
        teststat = None
        try:
            tvals = getattr(res, "tvalues", None)
            if tvals is None:
                tvals = getattr(res, "zvalues", None)
            if tvals is not None:
                teststat = float(tvals[var])
            else:
                # fallback compute
                if se and se != 0:
                    teststat = coef / se
        except Exception:
            teststat = None

        return {
            "coef": coef,
            "se": se,
            "teststat": teststat,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    def interpret_ols_logcoef(stats):
        """Interpret coef from OLS where DV = log(1 + Fatalities)."""
        if stats is None:
            return None
        coef = stats["coef"]
        # multiplicative change in (1 + Fatalities)
        try:
            mult = math.exp(coef)
            pct_change = (mult - 1.0) * 100.0
        except Exception:
            mult = None
            pct_change = None
        return {
            "coef": coef,
            "se": stats["se"],
            "pvalue": stats["pvalue"],
            "t_or_z": stats["teststat"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "multiplicative_change_in_1_plus_fatalities": mult,
            "percent_change_in_1_plus_fatalities": pct_change
        }

    def interpret_countcoef(stats):
        """Interpret coef from count GLM: log(expected Fatalities) coefficient."""
        if stats is None:
            return None
        coef = stats["coef"]
        try:
            mult = math.exp(coef)
            pct_change = (mult - 1.0) * 100.0
        except Exception:
            mult = None
            pct_change = None
        return {
            "coef": coef,
            "se": stats["se"],
            "pvalue": stats["pvalue"],
            "z_or_t": stats["teststat"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "multiplicative_change_in_expected_count": mult,
            "percent_change_in_expected_count": pct_change
        }

    # Prepare outputs
    out = {"masfem_ols": None, "female_bin_ols": None, "count_model": None}
    reasons = []

    # Extract OLS on log fatalities with MasFem_z
    ols_masfem = model_output.get("ols_logfatalities_masfem")
    stats_masfem = safe_get_stats(ols_masfem, "MasFem_z")
    out["masfem_ols"] = interpret_ols_logcoef(stats_masfem)

    # Extract OLS on log fatalities with Female binary
    ols_female = model_output.get("ols_logfatalities_femalebin")
    stats_female = safe_get_stats(ols_female, "Female")
    out["female_bin_ols"] = interpret_ols_logcoef(stats_female)

    # Extract count model (Negative Binomial or Poisson) on raw Fatalities
    count_model = model_output.get("count_model_fatalities_nb_or_poisson")
    stats_count = safe_get_stats(count_model, "MasFem_z")
    out["count_model"] = interpret_countcoef(stats_count)

    # Build concise interpretation for the hypothesis:
    # Hypothesis: more feminine names -> perceived less threatening -> fewer precautions -> MORE fatalities.
    # So we expect a POSITIVE association between femininity and fatalities.
    def assess_direction_and_significance(entry, name, expected_positive=True):
        if entry is None:
            return f"{name}: no result available."
        coef = entry.get("coef")
        p = entry.get("pvalue")
        if coef is None:
            return f"{name}: coefficient missing."
        sign = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        signif = None
        if p is None:
            signif = "p-value not available"
        else:
            if p < 0.05:
                signif = f"statistically significant (p = {p:.3g})"
            elif p < 0.1:
                signif = f"weak evidence (p = {p:.3g})"
            else:
                signif = f"not statistically significant (p = {p:.3g})"
        # Determine support: expected_positive True means positive coef supports hypothesis
        supports = False
        if p is not None and p < 0.05:
            supports = (coef > 0) if expected_positive else (coef < 0)
        elif p is not None and p < 0.1:
            supports = (coef > 0) if expected_positive else (coef < 0)
            # mark as weak
        else:
            supports = False
        support_text = "supports" if supports else "does not provide statistically significant support for"
        return f"{name}: coefficient {coef:+.3f} ({sign}), {signif}; this {support_text} the hypothesis."

    # For our advertised hypothesis, we expect a POSITIVE coefficient (more feminine -> more fatalities).
    assessments = [
        assess_direction_and_significance(out["masfem_ols"], "OLS (LogFatalities ~ MasFem_z)", expected_positive=True),
        assess_direction_and_significance(out["female_bin_ols"], "OLS (LogFatalities ~ Female)", expected_positive=True),
        assess_direction_and_significance(out["count_model"], "Count model (Fatalities ~ MasFem_z)", expected_positive=True)
    ]

    # Compose description
    description_lines = [
        "Extracted model summaries for the focal IVs (MasFem_z and Female):",
        "",
        assessments[0],
        assessments[1],
        assessments[2],
        "",
        "Notes:",
        "- For OLS models with LogFatalities = log(1 + Fatalities), coefficient c implies multiplicative change in (1 + Fatalities) of exp(c).",
        "- For the count GLM, coefficient is on log(expected count); exp(coef) is the multiplicative effect on expected fatalities.",
        "- 'supports' is used only when coefficient sign aligns with the hypothesized direction and p < 0.05 (or weak evidence if 0.05 <= p < 0.1)."
    ]
    description = "\n".join(description_lines)

    return {"object": out, "description": description}