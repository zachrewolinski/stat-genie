def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of hurricane name femininity (name_fem_z)
    on log_deaths from a fitted statsmodels OLS RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": a dict of extracted numeric results (coefficients, SEs, p-values,
                  95% CIs, and simple slopes of name_fem_z at saffir_cat_z = -1, 0, +1).
      - "description": a human-readable interpretation of those statistics in the
                       context of the research question.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Verify attributes
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels result (missing .params)")

    params = res.params
    pvalues = res.pvalues
    ci = res.conf_int()
    cov = res.cov_params()
    df_resid = float(res.df_resid)  # residual degrees of freedom

    def _get_term(name):
        """Helper to safely fetch term info; returns dict or None if missing."""
        if name in params.index:
            coef = float(params[name])
            se = float(res.bse[name])
            p = float(pvalues[name])
            ci_lo, ci_hi = float(ci.loc[name, 0]), float(ci.loc[name, 1])
            return {"coef": coef, "se": se, "p": p, "ci": (ci_lo, ci_hi)}
        return None

    # Main effect term
    name_term = _get_term("name_fem_z")
    if name_term is None:
        raise KeyError("Term 'name_fem_z' not found in model parameters.")

    # Interaction term (handle possible ordering)
    interaction_name = None
    for candidate in ("name_fem_z:saffir_cat_z", "saffir_cat_z:name_fem_z"):
        if candidate in params.index:
            interaction_name = candidate
            break
    interaction_term = _get_term(interaction_name) if interaction_name is not None else None

    # Compute simple slopes of name_fem_z at saffir_cat_z = -1, 0, +1 (standardized units)
    simple_slopes = {}
    t_crit = stats.t.ppf(1 - 0.025, df_resid)  # two-sided 95% critical t

    # Covariance elements needed if interaction exists
    if interaction_term is not None:
        cov_nn = float(cov.loc["name_fem_z", "name_fem_z"])
        cov_ii = float(cov.loc[interaction_name, interaction_name])
        cov_ni = float(cov.loc["name_fem_z", interaction_name])
    else:
        cov_nn = float(cov.loc["name_fem_z", "name_fem_z"])
        cov_ii = None
        cov_ni = None

    for s in (-1.0, 0.0, 1.0):
        slope = name_term["coef"] + (interaction_term["coef"] if interaction_term is not None else 0.0) * s
        # variance of linear combination: Var(b_name + s*b_int) = Var(b_name) + s^2 Var(b_int) + 2*s Cov(b_name,b_int)
        if interaction_term is not None:
            var = cov_nn + (s ** 2) * cov_ii + 2 * s * cov_ni
        else:
            var = cov_nn
        se = float(np.sqrt(var)) if var >= 0 else float(np.nan)
        tstat = slope / se if se and not np.isnan(se) else float("nan")
        p_two = float(2 * (1 - stats.t.cdf(abs(tstat), df_resid))) if not np.isnan(tstat) else float("nan")
        ci_lo = slope - t_crit * se
        ci_hi = slope + t_crit * se

        simple_slopes[s] = {
            "saffir_cat_z": s,
            "slope": float(slope),
            "se": float(se),
            "t": float(tstat),
            "p": float(p_two),
            "95%CI": (float(ci_lo), float(ci_hi)),
        }

    # Prepare return object
    result_object = {
        "name_term": name_term,
        "interaction_term": interaction_term,
        "simple_slopes": simple_slopes,
        "df_resid": float(df_resid),
    }

    # Prepare human-readable description
    # Summarize main effect and interaction
    def fmt_term(t):
        return f"coef={t['coef']:.4f}, se={t['se']:.4f}, p={t['p']:.4f}, 95%CI=({t['ci'][0]:.4f}, {t['ci'][1]:.4f})"

    desc_lines = []
    desc_lines.append("Extracted statistics for the effect of name femininity (name_fem_z) on log_deaths.")
    desc_lines.append(f"Main effect (name_fem_z): {fmt_term(name_term)}.")
    if interaction_term is not None:
        desc_lines.append(f"Interaction (name_fem_z x saffir_cat_z): {fmt_term(interaction_term)}.")
        desc_lines.append(
            "Simple slopes (effect of a 1 SD increase in name femininity on log_deaths) "
            "are provided at saffir_cat_z = -1, 0, +1 (approx. 1 SD below mean, mean, 1 SD above mean)."
        )
        for s in (-1.0, 0.0, 1.0):
            ss = simple_slopes[s]
            sig = "p<0.05" if ss["p"] < 0.05 else f"p={ss['p']:.3f}"
            desc_lines.append(
                f"  saffir_cat_z={s:+.0f}: slope={ss['slope']:.4f}, se={ss['se']:.4f}, {sig}, 95%CI=({ss['95%CI'][0]:.4f}, {ss['95%CI'][1]:.4f})"
            )
    else:
        desc_lines.append("No interaction term between name_fem_z and saffir_cat_z was found in the model.")
        ss = simple_slopes[0.0]
        sig = "p<0.05" if ss["p"] < 0.05 else f"p={ss['p']:.3f}"
        desc_lines.append(
            f"Estimated main effect (constant across Saffir categories): slope={ss['slope']:.4f}, se={ss['se']:.4f}, {sig}, 95%CI=({ss['95%CI'][0]:.4f}, {ss['95%CI'][1]:.4f})"
        )

    # Final interpretation sentence relating to hypothesis:
    # Positive slope => more feminine names associated with higher log_deaths (i.e., fewer precautions)
    slope0 = simple_slopes[0.0]["slope"]
    p0 = simple_slopes[0.0]["p"]
    if np.isfinite(slope0):
        if p0 < 0.05:
            if slope0 > 0:
                desc_lines.append(
                    "Interpretation: There is a statistically significant positive association between name femininity and log_deaths (at saffir_cat_z=0), "
                    "meaning more feminine names are associated with higher fatalities (consistent with the hypothesis that feminine names lead to fewer precautions)."
                )
            else:
                desc_lines.append(
                    "Interpretation: There is a statistically significant negative association between name femininity and log_deaths (at saffir_cat_z=0), "
                    "meaning more feminine names are associated with lower fatalities (contrary to the hypothesis)."
                )
        else:
            desc_lines.append(
                "Interpretation: The association between name femininity and log_deaths is not statistically significant at conventional levels (p>=0.05)."
            )
    else:
        desc_lines.append("Interpretation: Could not compute a reliable slope (SE/variance issue).")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}