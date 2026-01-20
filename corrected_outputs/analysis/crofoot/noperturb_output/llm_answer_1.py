def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, odds ratios, and
    marginal (interaction-adjusted) effects from a statsmodels GLMResultsWrapper
    (logistic regression) fitted with the formula:
        win ~ size_diff_z * LocAdv_z + male_diff_z + female_diff_z + C(dyad)

    Returns:
        dict with keys:
            - "object": dict containing coefficient table for focal terms and
                        marginal effects of size_diff_z at LocAdv_z = -1, 0, +1 (z-units).
            - "description": plain-language summary interpreting the direction
                             and significance of effects (alpha = 0.05).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Ensure the object looks like a fitted statsmodels results object
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels Results object.")

    # Possible parameter names for the interaction in statsmodels: 'size_diff_z:LocAdv_z'
    term_names = {
        "size": "size_diff_z",
        "loc": "LocAdv_z",
        "int": "size_diff_z:LocAdv_z"
    }

    # If interaction named differently (rare), try swapping order
    if term_names["int"] not in res.params.index:
        alt_int = "LocAdv_z:size_diff_z"
        if alt_int in res.params.index:
            term_names["int"] = alt_int

    # Collect coefficient table for focal terms (if present)
    def safe_get(name):
        if name in res.params.index:
            coef = float(res.params[name])
            se = float(res.bse[name])
            p = float(res.pvalues[name])
            ci_low, ci_high = res.conf_int().loc[name].astype(float)
            or_val = float(np.exp(coef))
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
            return {
                "coef": coef,
                "std_err": se,
                "z_or_wald": coef / se if se != 0 else None,
                "p_value": p,
                "ci_95": (ci_low, ci_high),
                "odds_ratio": or_val,
                "odds_ratio_ci_95": or_ci
            }
        else:
            return None

    size_tbl = safe_get(term_names["size"])
    loc_tbl = safe_get(term_names["loc"])
    int_tbl = safe_get(term_names["int"])

    # Prepare marginal effect of size_diff_z at specific LocAdv_z values (z = -1, 0, +1).
    # marginal effect (log-odds change per 1 SD increase in size) = beta_size + beta_interaction * loc_value
    marg_effects = {}
    cov = res.cov_params()
    # Try to obtain covariances needed for delta method; if missing, set se to None
    for loc_value in [-1.0, 0.0, 1.0]:
        label = f"size_effect_at_LocAdv_z={loc_value:+.0f}"
        if (term_names["size"] in res.params.index) and (term_names["int"] in res.params.index):
            b_size = float(res.params[term_names["size"]])
            b_int = float(res.params[term_names["int"]])
            est_logodds = b_size + b_int * loc_value

            # delta method variance: Var(b_size + loc*b_int) = Var(b_size) + loc^2 Var(b_int) + 2*loc*Cov(b_size,b_int)
            try:
                var_b_size = float(cov.loc[term_names["size"], term_names["size"]])
                var_b_int = float(cov.loc[term_names["int"], term_names["int"]])
                cov_bs_bi = float(cov.loc[term_names["size"], term_names["int"]])
                var_marg = var_b_size + (loc_value ** 2) * var_b_int + 2 * loc_value * cov_bs_bi
                se_marg = float(np.sqrt(var_marg)) if var_marg >= 0 else None
                ci_low_log = est_logodds - 1.96 * se_marg if se_marg is not None else None
                ci_high_log = est_logodds + 1.96 * se_marg if se_marg is not None else None
                or_est = float(np.exp(est_logodds))
                or_ci = (float(np.exp(ci_low_log)), float(np.exp(ci_high_log))) if ci_low_log is not None else (None, None)
            except Exception:
                se_marg = None
                or_est = float(np.exp(est_logodds))
                or_ci = (None, None)

            marg_effects[label] = {
                "loc_value": loc_value,
                "log_odds_change_per_1SD_size": est_logodds,
                "se_log_odds": se_marg,
                "log_odds_ci_95": (ci_low_log, ci_high_log) if se_marg is not None else (None, None),
                "odds_ratio_per_1SD_size": or_est,
                "odds_ratio_ci_95": or_ci
            }
        elif term_names["size"] in res.params.index:
            # No interaction present; marginal effect is simply the size coefficient (same for all loc)
            b_size = float(res.params[term_names["size"]])
            est_logodds = b_size
            try:
                se_marg = float(res.bse[term_names["size"]])
                ci_low_log = est_logodds - 1.96 * se_marg
                ci_high_log = est_logodds + 1.96 * se_marg
                or_est = float(np.exp(est_logodds))
                or_ci = (float(np.exp(ci_low_log)), float(np.exp(ci_high_log)))
            except Exception:
                se_marg = None
                or_est = float(np.exp(est_logodds))
                or_ci = (None, None)
            marg_effects[label] = {
                "loc_value": loc_value,
                "log_odds_change_per_1SD_size": est_logodds,
                "se_log_odds": se_marg,
                "log_odds_ci_95": (ci_low_log, ci_high_log) if se_marg is not None else (None, None),
                "odds_ratio_per_1SD_size": or_est,
                "odds_ratio_ci_95": or_ci
            }
        else:
            marg_effects[label] = None

    # Create a concise description based on p-values/signs of focal terms
    def sig_label(p):
        if p is None:
            return "NA"
        if p < 0.001:
            return "p < 0.001"
        elif p < 0.01:
            return "p < 0.01"
        elif p < 0.05:
            return "p < 0.05"
        else:
            return f"p = {p:.3f}"

    desc_lines = []
    desc_lines.append("Extracted focal-term estimates from the logistic model (effect on log-odds of focal group winning):")

    if size_tbl is not None:
        desc_lines.append(
            f"- Relative group size (size_diff_z): coef = {size_tbl['coef']:.3f}, "
            f"SE = {size_tbl['std_err']:.3f}, {sig_label(size_tbl['p_value'])}. "
            f"OR per 1 SD = {size_tbl['odds_ratio']:.3f} (95% CI: {size_tbl['odds_ratio_ci_95'][0]:.3f}, {size_tbl['odds_ratio_ci_95'][1]:.3f})."
        )
    else:
        desc_lines.append("- Relative group size (size_diff_z): term not found in model output.")

    if loc_tbl is not None:
        desc_lines.append(
            f"- Location advantage (LocAdv_z): coef = {loc_tbl['coef']:.3f}, "
            f"SE = {loc_tbl['std_err']:.3f}, {sig_label(loc_tbl['p_value'])}. "
            f"OR per 1 SD = {loc_tbl['odds_ratio']:.3f} (95% CI: {loc_tbl['odds_ratio_ci_95'][0]:.3f}, {loc_tbl['odds_ratio_ci_95'][1]:.3f})."
        )
    else:
        desc_lines.append("- Location advantage (LocAdv_z): term not found in model output.")

    if int_tbl is not None:
        desc_lines.append(
            f"- Interaction (size_diff_z:LocAdv_z): coef = {int_tbl['coef']:.3f}, "
            f"SE = {int_tbl['std_err']:.3f}, {sig_label(int_tbl['p_value'])}. "
            f"Interpretation: a {'positive' if int_tbl['coef']>0 else 'negative'} interaction indicates that the effect of relative group size on winning "
            f"{'increases' if int_tbl['coef']>0 else 'decreases'} as location advantage increases."
        )
        # Add marginal effect summary
        desc_lines.append("- Marginal effect of relative size (per 1 SD) at representative LocAdv_z values:")
        for k, v in marg_effects.items():
            if v is None:
                desc_lines.append(f"  * {k}: not available")
            else:
                se_text = f"SE(log-odds) = {v['se_log_odds']:.3f}" if v['se_log_odds'] is not None else "SE unavailable"
                ci_or = v["odds_ratio_ci_95"]
                ci_text = f"OR = {v['odds_ratio_per_1SD_size']:.3f} (95% CI: {ci_or[0]:.3f}, {ci_or[1]:.3f})" if ci_or[0] is not None else f"OR = {v['odds_ratio_per_1SD_size']:.3f}"
                desc_lines.append(f"  * LocAdv_z={v['loc_value']:+.0f}: log-odds change = {v['log_odds_change_per_1SD_size']:.3f}; {se_text}; {ci_text}")
    else:
        desc_lines.append("- Interaction term not present or not estimable. Marginal effect of size is constant across location advantage (see size_diff_z above).")

    # Final concise answer about influence
    # Decide significance labels
    size_sig = size_tbl and (size_tbl["p_value"] < 0.05)
    loc_sig = loc_tbl and (loc_tbl["p_value"] < 0.05)
    int_sig = int_tbl and (int_tbl["p_value"] < 0.05)

    final_sent = "Summary conclusion: "
    parts = []
    if int_tbl is not None and int_sig:
        parts.append("There is a statistically significant interaction between relative group size and location advantage (p < 0.05).")
        parts.append("This means the effect of relative group size on winning depends on contest location: use the marginal effects above to see how the size effect changes at different location advantages.")
    else:
        # No significant interaction
        if size_tbl is not None:
            if size_sig:
                parts.append("Relative group size is a significant predictor: larger focal groups have higher probability of winning (positive coef).")
            else:
                parts.append("Relative group size does not show a statistically significant effect at alpha=0.05.")
        if loc_tbl is not None:
            if loc_sig:
                parts.append("Location advantage is a significant predictor: groups closer to home are more likely to win.")
            else:
                parts.append("Location advantage does not show a statistically significant effect at alpha=0.05.")
        if (not size_tbl) and (not loc_tbl):
            parts.append("Model did not contain the expected focal terms or they could not be extracted.")

    final_sent += " ".join(parts)
    desc_lines.append(final_sent)

    # Construct object to return: include tables and marginal effects
    out_obj = {
        "coefficients": {
            "size_diff_z": size_tbl,
            "LocAdv_z": loc_tbl,
            "size_diff_z:LocAdv_z": int_tbl
        },
        "marginal_effects_of_size_at_LocAdv_z": marg_effects,
        "model_summary_str": res.summary().as_text() if hasattr(res, "summary") else None
    }

    return {
        "object": out_obj,
        "description": "\n".join(desc_lines)
    }