def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, odds ratios, and
    tests whether the effect of relative group size (log_rel_size) on win
    probability differs by contest Location (interaction terms). Returns a
    dictionary ("object") with detailed stats and a brief "description".
    """
    import re
    import math
    import numpy as np
    import pandas as pd

    # Helper: two-sided p-value from z using math.erf (no scipy dependency)
    def p_from_z(z):
        return math.erfc(abs(z) / math.sqrt(2))

    # Accept either a statsmodels results object or a wrapper from get_robustcov_results
    res = model_output

    # Ensure required attributes exist
    if not (hasattr(res, "params") and hasattr(res, "bse") and hasattr(res, "pvalues")):
        raise ValueError("model_output does not look like a statsmodels results object "
                         "(missing params, bse or pvalues).")

    params = pd.Series(res.params)
    bse = pd.Series(res.bse) if hasattr(res, "bse") else None
    pvalues = pd.Series(res.pvalues) if hasattr(res, "pvalues") else None

    # Confidence intervals (may be method-adjusted if robust cov used)
    try:
        conf = res.conf_int()
        conf = pd.DataFrame(conf)
        conf.columns = ["ci_lower", "ci_upper"]
    except Exception:
        # If conf_int unavailable, compute approx using bse
        if bse is not None:
            conf = pd.DataFrame({
                "ci_lower": params - 1.96 * bse,
                "ci_upper": params + 1.96 * bse
            })
        else:
            conf = pd.DataFrame({"ci_lower": params * np.nan, "ci_upper": params * np.nan})

    # Covariance matrix (needed for linear combinations / interaction tests)
    try:
        cov = res.cov_params()
        cov = pd.DataFrame(cov)
    except Exception:
        cov = None

    # Build term-level summary for terms of interest
    terms_of_interest = [name for name in params.index
                         if ("log_rel_size" in name) or ("C(Location)" in name) or (":Location" in name)]
    # Also include the main Location terms explicitly (common name pattern: C(Location)[T.level])
    location_terms = [name for name in params.index if "C(Location)" in name and "log_rel_size" not in name]

    summary = {}
    for name in terms_of_interest:
        coef = float(params.get(name, np.nan))
        se = float(bse.get(name, np.nan)) if bse is not None else np.nan
        p = float(pvalues.get(name, np.nan)) if pvalues is not None else np.nan
        ci_low = float(conf.loc[name, "ci_lower"]) if name in conf.index else np.nan
        ci_high = float(conf.loc[name, "ci_upper"]) if name in conf.index else np.nan
        or_est = math.exp(coef) if not np.isnan(coef) else np.nan
        or_ci = (math.exp(ci_low) if not np.isnan(ci_low) else np.nan,
                 math.exp(ci_high) if not np.isnan(ci_high) else np.nan)

        summary[name] = {
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_coef_95": (ci_low, ci_high),
            "odds_ratio": or_est,
            "ci_odds_ratio_95": or_ci
        }

    # Compute marginal (slope) effect of log_rel_size at each Location level.
    # Strategy:
    # - Try to get original Location levels from the model data frame (if available)
    # - If not available, infer levels from parameter names (C(Location)[T.<level>])
    levels = []
    base_level = None
    try:
        df = res.model.data.frame
        if "Location" in df.columns:
            # preserve order as in the data (first appearance)
            levels = list(pd.Categorical(df["Location"]).categories)
    except Exception:
        pass

    # Infer levels from parameter names if needed
    infer_levels = []
    for name in params.index:
        m = re.search(r"C\(Location\)\[T\.([^\]]+)\]", name)
        if m:
            infer_levels.append(m.group(1))
    infer_levels = list(dict.fromkeys(infer_levels))  # unique preserve order

    if levels == [] and infer_levels:
        # We have only non-reference levels in infer_levels; need to set base_level unknown
        # We will set levels = infer_levels + ['<reference>'] and treat reference specially.
        levels = infer_levels + ["<reference>"]
        # Determine which levels have explicit params; the one missing is reference
        base_level = "<reference>"
    else:
        # Try to determine reference/base level by seeing which level is missing from C(Location) params
        if levels:
            # see which of levels has no explicit C(Location)[T.level] param
            found = set(infer_levels)
            missing = [lev for lev in levels if lev not in found]
            if len(missing) == 1:
                base_level = missing[0]
            elif len(missing) >= 1:
                # If multiple missing (unlikely), pick the first as base
                base_level = missing[0]
            else:
                base_level = None

    # Identify the main coefficient for log_rel_size
    if "log_rel_size" in params.index:
        base_log_coef = float(params["log_rel_size"])
    else:
        # maybe the term got renamed (rare). Try to find param name equal to 'log_rel_size'
        candidate = [n for n in params.index if n.endswith("log_rel_size")]
        base_log_coef = float(params[candidate[0]]) if candidate else np.nan

    marginal_effects = {}
    for lev in levels:
        if lev == base_level:
            # slope is simply base_log_coef
            slope = base_log_coef
            # variance is var(log_rel_size)
            if cov is not None and "log_rel_size" in cov.index:
                var = float(cov.loc["log_rel_size", "log_rel_size"])
            else:
                var = float(bse["log_rel_size"] ** 2) if (bse is not None and "log_rel_size" in bse.index) else np.nan
            se_slope = math.sqrt(var) if not np.isnan(var) else np.nan
            z = slope / se_slope if (not np.isnan(slope) and not np.isnan(se_slope) and se_slope != 0) else np.nan
            p_slope = p_from_z(z) if not np.isnan(z) else np.nan
            ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan
        else:
            # Look for interaction term name patterns
            # pattern 1: 'log_rel_size:C(Location)[T.<lev>]'
            pat1 = f"log_rel_size:C(Location)[T.{lev}]"
            pat2 = f"C(Location)[T.{lev}]:log_rel_size"
            inter_name = None
            if pat1 in params.index:
                inter_name = pat1
            elif pat2 in params.index:
                inter_name = pat2
            else:
                # try variations where patsy might have sanitized the level string
                matches = [n for n in params.index if ("log_rel_size" in n and f"T.{lev}" in n)]
                if matches:
                    inter_name = matches[0]

            if inter_name is None:
                # no explicit interaction term found -> slope same as base
                slope = base_log_coef
                # variance same as base
                if cov is not None and "log_rel_size" in cov.index:
                    var = float(cov.loc["log_rel_size", "log_rel_size"])
                else:
                    var = float(bse["log_rel_size"] ** 2) if (bse is not None and "log_rel_size" in bse.index) else np.nan
                se_slope = math.sqrt(var) if not np.isnan(var) else np.nan
                z = slope / se_slope if (not np.isnan(slope) and not np.isnan(se_slope) and se_slope != 0) else np.nan
                p_slope = p_from_z(z) if not np.isnan(z) else np.nan
                ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
                ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            else:
                inter_coef = float(params.get(inter_name, 0.0))
                slope = base_log_coef + inter_coef
                # variance of sum = var(A)+var(B)+2cov(A,B)
                if cov is not None and "log_rel_size" in cov.index and inter_name in cov.index:
                    var = float(cov.loc["log_rel_size", "log_rel_size"] +
                                cov.loc[inter_name, inter_name] +
                                2 * cov.loc["log_rel_size", inter_name])
                else:
                    # fallback: approximate using available bse (assume cov=0) -> conservative/approximate
                    varA = float(bse["log_rel_size"] ** 2) if (bse is not None and "log_rel_size" in bse.index) else np.nan
                    varB = float(bse[inter_name] ** 2) if (bse is not None and inter_name in bse.index) else np.nan
                    if not np.isnan(varA) and not np.isnan(varB):
                        var = varA + varB
                    else:
                        var = np.nan
                se_slope = math.sqrt(var) if not np.isnan(var) else np.nan
                z = slope / se_slope if (not np.isnan(slope) and not np.isnan(se_slope) and se_slope != 0) else np.nan
                p_slope = p_from_z(z) if not np.isnan(z) else np.nan
                ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
                ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan

        marginal_effects[str(lev)] = {
            "slope_log_rel_size": slope,
            "se_slope": se_slope,
            "p_value_slope": p_slope,
            "ci_coef_95": (ci_low, ci_high),
            "odds_ratio_per_unit_log_rel_size": math.exp(slope) if not np.isnan(slope) else np.nan,
            "ci_odds_ratio_95": (math.exp(ci_low) if not np.isnan(ci_low) else np.nan,
                                 math.exp(ci_high) if not np.isnan(ci_high) else np.nan)
        }

    # Package final object
    result_object = {
        "term_summary": summary,
        "location_main_terms": {lt: summary.get(lt, None) for lt in location_terms},
        "marginal_effects_by_Location": marginal_effects,
        "notes": {
            "interpretation_guideline": (
                "Positive coefficient for log_rel_size => higher relative group size (focal vs other) "
                "increases the odds that the focal group wins. The exponential of a coefficient is an odds ratio: "
                "values >1 mean higher odds of focal win per unit increase in the predictor. "
                "Interaction terms (log_rel_size:C(Location)[T.level]) indicate how the slope of log_rel_size "
                "differs at that Location level compared to the reference level. "
                "For the slope of log_rel_size at each Location we provide coef, se, p-value, 95% CI, and OR."
            ),
            "stat_method": (
                "Coefficients, p-values and CIs are taken from the provided statsmodels results object. "
                "For testing the slope of log_rel_size in each Location, variance of linear combinations is "
                "computed using the model covariance matrix if available (gives correct robust clustered SE if "
                "the provided result is a robust results object). If covariance is not available, an approximate "
                "fallback using individual bse is used."
            )
        }
    }

    description = (
        "Returned object includes: (1) coefficient, SE, p-value, 95% CI, and odds ratio for each model term "
        "involving log_rel_size and Location; (2) the marginal (slope) effect of log_rel_size computed for each "
        "Location level (this shows how the effect of relative group size on winning probability differs by location). "
        "Interpretation note: a positive slope_log_rel_size means that being relatively larger increases the probability "
        "that the focal group wins; the interaction terms reveal whether that positive effect is stronger or weaker "
        "when the contest is closer to either group's home-range or on the boundary."
    )

    return {"object": result_object, "description": description}