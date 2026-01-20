def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression)
    that tested: Win ~ z_RelativeGroupSize * z_DistanceDifference + z_MaleDiff + z_FemaleDiff + C(DyadID)
    
    Returns a dictionary with:
      - "object": a dict of extracted numeric results for the main terms and their interpretation-ready summaries
      - "description": a short plain-language interpretation of what these statistics imply about
                       how relative group size and contest location influence the probability of winning.
    """
    import numpy as np
    from math import exp
    
    res = model_output  # expected statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
    
    # Helpers to robustly find term names (interaction may be ':' or ':')
    params = getattr(res, "params")
    pvalues = getattr(res, "pvalues")
    bse = getattr(res, "bse")
    try:
        ci_df = res.conf_int()
    except Exception:
        # fallback: construct simple CI if conf_int not available
        z = 1.96
        ci_df = None
    
    def get_term_stats(term_candidates):
        """Search possible term names and return stats dict or NaNs if not found."""
        for term in term_candidates:
            if term in params.index:
                coef = float(params.loc[term])
                se = float(bse.loc[term]) if term in bse.index else float(np.nan)
                pval = float(pvalues.loc[term]) if term in pvalues.index else float(np.nan)
                if ci_df is not None and term in ci_df.index:
                    ci_low, ci_high = float(ci_df.loc[term, 0]), float(ci_df.loc[term, 1])
                else:
                    # approximate 95% CI from coef +/- 1.96*se if se available
                    if not np.isnan(se):
                        ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se
                    else:
                        ci_low, ci_high = float(np.nan), float(np.nan)
                return {
                    "term_name": term,
                    "coef": coef,
                    "se": se,
                    "p_value": pval,
                    "ci_95": (ci_low, ci_high),
                    "odds_ratio": float(np.exp(coef)) if not np.isnan(coef) else float(np.nan),
                    "odds_ratio_95": (float(np.exp(ci_low)) if not np.isnan(ci_low) else float(np.nan),
                                      float(np.exp(ci_high)) if not np.isnan(ci_high) else float(np.nan)),
                }
        # not found
        return {
            "term_name": None,
            "coef": float(np.nan),
            "se": float(np.nan),
            "p_value": float(np.nan),
            "ci_95": (float(np.nan), float(np.nan)),
            "odds_ratio": float(np.nan),
            "odds_ratio_95": (float(np.nan), float(np.nan)),
        }
    
    # Candidate names for the main predictors and interaction:
    rel_size_stats = get_term_stats(["z_RelativeGroupSize", "z_RelativeGroupSize"])
    dist_stats = get_term_stats(["z_DistanceDifference", "z_DistanceDifference"])
    inter_stats = get_term_stats([
        "z_RelativeGroupSize:z_DistanceDifference",
        "z_DistanceDifference:z_RelativeGroupSize",
        "z_RelativeGroupSize*z_DistanceDifference",  # unlikely, but safe
        "z_RelativeGroupSize:z_DistanceDifference"
    ])
    
    # Intercept name can be 'Intercept' or 'const'
    intercept_name = None
    for n in ["Intercept", "const"]:
        if n in params.index:
            intercept_name = n
            break
    if intercept_name is not None:
        intercept = float(params.loc[intercept_name])
    else:
        intercept = float(np.nan)
    
    # Compute marginal effects of a +1 SD change in relative group size at different location z-values
    # marginal_logit = beta_rel + beta_inter * z_dist
    beta_rel = rel_size_stats["coef"]
    beta_dist = dist_stats["coef"]
    beta_int = inter_stats["coef"]
    
    def logistic(x):
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except Exception:
            return float(np.nan)
    
    # Baseline probability at intercept (other predictors set to 0; dyad FE ignored / assumed zero-level)
    baseline_prob = logistic(intercept) if not np.isnan(intercept) else float(np.nan)
    
    # Compute marginal effects at z_distance = 0 (mean), +1, -1
    margins = {}
    for z_val in [0.0, 1.0, -1.0]:
        if not np.isnan(beta_rel):
            marginal_logit = beta_rel + (beta_int * z_val if not np.isnan(beta_int) else 0.0)
            # log-odds change for +1 SD relative group size at given z_dist
            odds_ratio = float(np.exp(marginal_logit))
            # Change in probability for +1 SD in rel size from baseline intercept:
            if not np.isnan(intercept):
                p_before = baseline_prob
                p_after = logistic(intercept + marginal_logit)
                prob_change = p_after - p_before
            else:
                p_before = float(np.nan)
                p_after = float(np.nan)
                prob_change = float(np.nan)
            margins[f"zdist_{z_val}"] = {
                "z_distance": z_val,
                "marginal_logit_per_1sd_rel_size": marginal_logit,
                "odds_ratio_per_1sd_rel_size": odds_ratio,
                "probability_change_from_intercept_for_+1sd_rel_size": prob_change,
                "prob_before": p_before,
                "prob_after": p_after,
            }
        else:
            margins[f"zdist_{z_val}"] = {
                "z_distance": z_val,
                "marginal_logit_per_1sd_rel_size": float(np.nan),
                "odds_ratio_per_1sd_rel_size": float(np.nan),
                "probability_change_from_intercept_for_+1sd_rel_size": float(np.nan),
                "prob_before": float(np.nan),
                "prob_after": float(np.nan),
            }
    
    # Build concise conclusion statements about statistical significance and direction
    def significance_statement(term_stats, label):
        p = term_stats["p_value"]
        if np.isnan(p):
            return f"{label}: no estimate available."
        sig = p < 0.05
        direction = "positive" if term_stats["coef"] > 0 else ("negative" if term_stats["coef"] < 0 else "null")
        return (f"{label}: coef={term_stats['coef']:.3f}, p={p:.3f} "
                f"({'significant' if sig else 'ns'}, {direction}), odds-ratio={term_stats['odds_ratio']:.3f}, "
                f"95% CI coef={term_stats['ci_95'][0]:.3f} to {term_stats['ci_95'][1]:.3f}")
    
    rel_stmt = significance_statement(rel_size_stats, "Relative group size (z_RelativeGroupSize)")
    dist_stmt = significance_statement(dist_stats, "Contest location (z_DistanceDifference)")
    int_stmt = significance_statement(inter_stats, "Interaction (RelativeSize x Distance)")
    
    # Compose return object
    out_object = {
        "terms": {
            "relative_group_size": rel_size_stats,
            "distance_difference": dist_stats,
            "interaction": inter_stats,
            "intercept": {
                "name": intercept_name,
                "value": intercept
            }
        },
        "marginal_effects_of_rel_size_at_distance_z": margins,
        "baseline_probability_at_intercept": baseline_prob,
        "statements": {
            "relative": rel_stmt,
            "distance": dist_stmt,
            "interaction": int_stmt
        }
    }
    
    # Short plain-language description summarizing what the stats imply
    # This will say whether focal group size increases winning odds, whether location matters, and whether there's moderation.
    # Rely on p-values < 0.05 to call "statistically significant"
    def interpret_term(term_stats, label):
        p = term_stats["p_value"]
        if np.isnan(p):
            return f"{label}: estimate not available."
        if p < 0.05:
            dir_word = "increases" if term_stats["coef"] > 0 else "decreases"
            return f"{label} {dir_word} the log-odds (and thus odds) of the focal group winning (coef={term_stats['coef']:.3f}, p={p:.3f})."
        else:
            return f"{label} shows no statistically significant effect (coef={term_stats['coef']:.3f}, p={p:.3f})."
    
    description_lines = [
        interpret_term(rel_size_stats, "Relative group size (focal - other)"),
        interpret_term(dist_stats, "Contest location (closer to focal home)"),
        interpret_term(inter_stats, "Interaction between relative size and location"),
        "Marginal effects provided show the change in log-odds, odds-ratio, and approximate probability change for a +1 SD increase in relative group size when the contest location z-distance is 0 (mean), +1, and -1."
    ]
    description = " ".join(description_lines)
    
    return {"object": out_object, "description": description}