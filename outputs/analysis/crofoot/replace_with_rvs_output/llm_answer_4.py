def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, 95% CIs, odds ratios and a short
    interpretation for the effects of:
      - rel_size_log (relative group size)
      - focal_home (home advantage)
      - rel_size_log:focal_home (interaction)

    Returns:
      {
        "object": { term_name: {coef, se, pvalue, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper, significant}, ...,
                   "marginal_rel_size_at_home0": {...}, "marginal_rel_size_at_home1": {...},
                   "nobs": int
                 }
        "description": short human-readable interpretation string
      }
    """
    import numpy as np
    import pandas as pd
    from math import exp, sqrt
    from scipy import stats

    res = model_output

    # Basic parameter table
    try:
        params = res.params  # pandas Series
        bse = res.bse
        pvals = res.pvalues
        conf = res.conf_int()  # DataFrame or ndarray
        cov = res.cov_params()
    except Exception as e:
        raise ValueError(f"Provided model_output does not expose expected attributes: {e}")

    # Ensure conf is DataFrame with index matching params
    if not isinstance(conf, pd.DataFrame):
        conf = pd.DataFrame(conf, index=params.index, columns=["ci_lower", "ci_upper"])
    else:
        # statsmodels sometimes returns columns named 0,1
        conf.columns = ["ci_lower", "ci_upper"]

    def safe_get(name_variants):
        """Return the exact parameter name found among params.index for any of the variants, else None."""
        for v in name_variants:
            if v in params.index:
                return v
        return None

    # Common possible names for the interaction (patsy uses ':' for numeric*numeric)
    term_rel = safe_get(["rel_size_log"])
    term_home = safe_get(["focal_home"])
    term_inter = safe_get(["rel_size_log:focal_home", "focal_home:rel_size_log", "rel_size_log*focal_home"])

    results = {}

    def make_term_entry(term_name):
        if term_name is None:
            return None
        coef = float(params[term_name])
        se = float(bse[term_name])
        p = float(pvals[term_name])
        ci_l = float(conf.loc[term_name, "ci_lower"])
        ci_u = float(conf.loc[term_name, "ci_upper"])
        OR = float(np.exp(coef))
        OR_ci_l = float(np.exp(ci_l))
        OR_ci_u = float(np.exp(ci_u))
        signif = bool(p < 0.05)
        return {
            "term": term_name,
            "coef": coef,
            "se": se,
            "pvalue": p,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "OR": OR,
            "OR_ci_lower": OR_ci_l,
            "OR_ci_upper": OR_ci_u,
            "significant": signif,
        }

    results["rel_size_log"] = make_term_entry(term_rel)
    results["focal_home"] = make_term_entry(term_home)
    results["interaction"] = make_term_entry(term_inter)

    # Marginal effect of rel_size_log when focal_home = 0 (just coef of rel_size_log)
    # and when focal_home = 1 (coef_rel + coef_inter). For the latter, compute SE using covariance.
    if term_rel is not None:
        rel_coef = float(params[term_rel])
        rel_se = float(bse[term_rel])
    else:
        rel_coef = None
        rel_se = None

    if term_inter is not None:
        inter_coef = float(params[term_inter])
    else:
        inter_coef = 0.0  # if no interaction term, marginal at home1 equals rel_coef

    # Marginal at home = 0
    if rel_coef is not None:
        me0_coef = rel_coef
        me0_se = rel_se
        me0_ci_l = me0_coef - 1.96 * me0_se
        me0_ci_u = me0_coef + 1.96 * me0_se
        me0_or = exp(me0_coef)
        me0_or_ci = (exp(me0_ci_l), exp(me0_ci_u))
        me0_p = float(pvals[term_rel])
        results["marginal_rel_size_at_home0"] = {
            "coef": me0_coef,
            "se": me0_se,
            "pvalue": me0_p,
            "ci_lower": me0_ci_l,
            "ci_upper": me0_ci_u,
            "OR": me0_or,
            "OR_ci_lower": me0_or_ci[0],
            "OR_ci_upper": me0_or_ci[1],
            "significant": bool(me0_p < 0.05),
        }
    else:
        results["marginal_rel_size_at_home0"] = None

    # Marginal at home = 1
    if rel_coef is not None:
        me1_coef = rel_coef + inter_coef
        # Compute variance of sum using cov matrix if possible
        try:
            var_rel = cov.loc[term_rel, term_rel]
            if term_inter is not None and term_inter in cov.index:
                var_inter = cov.loc[term_inter, term_inter]
                cov_rel_inter = cov.loc[term_rel, term_inter]
            else:
                var_inter = 0.0
                cov_rel_inter = 0.0
            me1_var = var_rel + var_inter + 2.0 * cov_rel_inter
            me1_se = float(np.sqrt(max(me1_var, 0.0)))
            me1_ci_l = me1_coef - 1.96 * me1_se
            me1_ci_u = me1_coef + 1.96 * me1_se
            me1_or = exp(me1_coef)
            me1_or_ci = (exp(me1_ci_l), exp(me1_ci_u))
            # approximate p-value for the sum using normal approx
            z = me1_coef / me1_se if me1_se > 0 else np.nan
            me1_p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if me1_se > 0 else np.nan
        except Exception:
            # fallback: cannot compute covariance; use naive sum of ses (conservative not correct)
            me1_se = None
            me1_ci_l = None
            me1_ci_u = None
            me1_or = exp(me1_coef)
            me1_or_ci = (None, None)
            me1_p = None

        results["marginal_rel_size_at_home1"] = {
            "coef": me1_coef,
            "se": me1_se,
            "pvalue": me1_p,
            "ci_lower": me1_ci_l,
            "ci_upper": me1_ci_u,
            "OR": me1_or,
            "OR_ci_lower": me1_or_ci[0],
            "OR_ci_upper": me1_or_ci[1],
            "significant": bool((me1_p is not None) and (me1_p < 0.05)),
        }
    else:
        results["marginal_rel_size_at_home1"] = None

    # Add sample size if available
    nobs = None
    try:
        nobs = int(res.nobs)
    except Exception:
        try:
            nobs = int(res.model.endog.shape[0])
        except Exception:
            nobs = None
    results["nobs"] = nobs

    # Short interpretation based on p-values
    conclusions = []
    if results["rel_size_log"] is not None:
        if results["rel_size_log"]["significant"]:
            conclusions.append("Relative group size (rel_size_log) has a statistically significant effect on the probability of winning (p < 0.05). Larger focal groups increase odds of winning (OR > 1 when coef > 0).")
        else:
            conclusions.append("Relative group size (rel_size_log) is not statistically significant at the 0.05 level.")
    else:
        conclusions.append("rel_size_log term not present in the model output.")

    if results["focal_home"] is not None:
        if results["focal_home"]["significant"]:
            conclusions.append("Being closer to the group's home center (focal_home) has a statistically significant effect on winning (p < 0.05), indicating a home advantage.")
        else:
            conclusions.append("focal_home is not statistically significant at the 0.05 level.")
    else:
        conclusions.append("focal_home term not present in the model output.")

    if results["interaction"] is not None:
        if results["interaction"]["significant"]:
            conclusions.append("The interaction between relative group size and home location is statistically significant, meaning the effect of size on winning differs depending on whether the focal group is at home.")
        else:
            conclusions.append("The interaction term is not statistically significant at the 0.05 level, suggesting the size effect does not change detectably with home status.")
    else:
        conclusions.append("No interaction term detected in the model output.")

    description = " ".join(conclusions)

    return {"object": results, "description": description}