def extract_final_answer(model_output):
    """
    Extracts the Reader View effect and its interaction with dyslexia from a fitted statsmodels OLS result.
    Returns a dict with keys:
      - "object": dict containing coefficients, p-values, 95% CIs, and percent-change interpretation
      - "description": brief interpretation answering whether Reader View improves reading speed,
                       separately for readers without dyslexia and with dyslexia.

    Expects model_output to be a statsmodels RegressionResults (or RegressionResultsWrapper).
    """
    import numpy as np

    res = model_output  # alias
    params = getattr(res, "params", None)
    pvalues = getattr(res, "pvalues", None)
    conf = None
    try:
        conf = res.conf_int()
    except Exception:
        conf = None

    # Names we expect
    main_name = "reader_view"
    interact_name = "reader_view:dyslexia_bin"
    dys_name = "dyslexia_bin"

    # Prepare output container
    out = {
        "reader_view": None,
        "dyslexia_bin": None,
        "interaction": None,
        "reader_view_effect_non_dyslexic": None,
        "reader_view_effect_dyslexic": None,
        "notes": []
    }

    # Helper to safely extract coef, pval, ci
    def get_coef_info(name):
        if params is None or name not in params.index:
            return None
        coef = float(params.loc[name])
        p = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
        ci = None
        if conf is not None and name in conf.index:
            ci = [float(conf.loc[name, 0]), float(conf.loc[name, 1])]
        return {"coef": coef, "pvalue": p, "ci_95": ci}

    out["reader_view"] = get_coef_info(main_name)
    out["dyslexia_bin"] = get_coef_info(dys_name)
    out["interaction"] = get_coef_info(interact_name)

    # Compute effect of reader_view for non-dyslexic (dyslexia_bin = 0): it's just main coef
    if out["reader_view"] is not None:
        coef_nd = out["reader_view"]["coef"]
        # percent change on original speed scale
        pct_nd = (np.exp(coef_nd) - 1) * 100.0
        ci_nd = None
        if out["reader_view"]["ci_95"] is not None:
            ci_nd = [ (np.exp(out["reader_view"]["ci_95"][0]) - 1) * 100.0,
                      (np.exp(out["reader_view"]["ci_95"][1]) - 1) * 100.0 ]
        out["reader_view_effect_non_dyslexic"] = {
            "coef_log_speed": coef_nd,
            "pvalue": out["reader_view"]["pvalue"],
            "ci_95_log_speed": out["reader_view"]["ci_95"],
            "pct_change_speed": pct_nd,
            "pct_change_speed_ci_95": ci_nd
        }
    else:
        out["notes"].append("Main effect coefficient 'reader_view' not found in model output.")

    # Compute effect of reader_view for dyslexic (dyslexia_bin = 1): main + interaction
    if out["reader_view"] is not None and out["interaction"] is not None:
        # Use model's t_test to get correct SE/pvalue/confint for linear combination
        try:
            # Build a string expression using exact parameter names
            expr = f"{main_name} + {interact_name} = 0"
            tt = res.t_test(expr)
            est = float(tt.effect.flatten()[0])
            p_comb = float(tt.pvalue) if hasattr(tt, "pvalue") else None
            ci_comb = None
            try:
                ci_mat = tt.conf_int()
                if ci_mat is not None:
                    ci_comb = [float(ci_mat[0, 0]), float(ci_mat[0, 1])]
            except Exception:
                ci_comb = None
            pct_comb = (np.exp(est) - 1) * 100.0
            pct_comb_ci = None
            if ci_comb is not None:
                pct_comb_ci = [ (np.exp(ci_comb[0]) - 1) * 100.0,
                                (np.exp(ci_comb[1]) - 1) * 100.0 ]
            out["reader_view_effect_dyslexic"] = {
                "coef_log_speed": est,
                "pvalue": p_comb,
                "ci_95_log_speed": ci_comb,
                "pct_change_speed": pct_comb,
                "pct_change_speed_ci_95": pct_comb_ci
            }
        except Exception as e:
            # Fallback to manual variance calculation if t_test fails
            try:
                cov = res.cov_params()
                v_main = cov.loc[main_name, main_name]
                v_int = cov.loc[interact_name, interact_name]
                cov_main_int = cov.loc[main_name, interact_name]
                est = out["reader_view"]["coef"] + out["interaction"]["coef"]
                var_sum = v_main + v_int + 2.0 * cov_main_int
                se_sum = float(np.sqrt(var_sum))
                df = getattr(res, "df_resid", None)
                t_stat = est / se_sum if se_sum != 0 else None
                # two-sided p-value using t approximation if df available, else fallback to None
                p_comb = None
                if t_stat is not None and df is not None:
                    try:
                        from scipy import stats
                        p_comb = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))
                    except Exception:
                        p_comb = None
                ci_lower = est - 1.96 * se_sum
                ci_upper = est + 1.96 * se_sum
                pct_comb = (np.exp(est) - 1) * 100.0
                pct_comb_ci = [ (np.exp(ci_lower) - 1) * 100.0, (np.exp(ci_upper) - 1) * 100.0 ]
                out["reader_view_effect_dyslexic"] = {
                    "coef_log_speed": float(est),
                    "pvalue": p_comb,
                    "ci_95_log_speed": [float(ci_lower), float(ci_upper)],
                    "pct_change_speed": pct_comb,
                    "pct_change_speed_ci_95": pct_comb_ci
                }
                out["notes"].append("Used manual variance combination fallback for dyslexic effect.")
            except Exception as e2:
                out["notes"].append("Could not compute combined effect for dyslexic readers: " + str(e2))
    else:
        out["notes"].append("Interaction or main effect missing, cannot compute dyslexic subgroup effect.")

    # Interpretation summary: determine statistical significance at alpha=0.05 if p-values are available
    def sig_text(p):
        if p is None:
            return "p-value unavailable"
        return ("statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)")

    interp_lines = []
    # Non-dyslexic
    nd = out.get("reader_view_effect_non_dyslexic")
    if nd is not None:
        interp_lines.append(
            f"For readers without dyslexia (dyslexia_bin=0): Reader View coef (log-speed) = {nd['coef_log_speed']:.4f}, "
            f"{sig_text(nd['pvalue'])}. This corresponds to a {nd['pct_change_speed']:.1f}% change in reading speed "
            f"(95% CI: {nd['pct_change_speed_ci_95'][0]:.1f}% to {nd['pct_change_speed_ci_95'][1]:.1f}%)"
            if nd['pct_change_speed_ci_95'] is not None else
            f"For readers without dyslexia (dyslexia_bin=0): Reader View coef (log-speed) = {nd['coef_log_speed']:.4f}, "
            f"{sig_text(nd['pvalue'])}."
        )
    else:
        interp_lines.append("No estimate available for non-dyslexic readers.")

    # Dyslexic
    d = out.get("reader_view_effect_dyslexic")
    if d is not None:
        interp_lines.append(
            f"For readers with dyslexia (dyslexia_bin=1): Reader View combined coef (log-speed) = {d['coef_log_speed']:.4f}, "
            f"{sig_text(d['pvalue'])}. This corresponds to a {d['pct_change_speed']:.1f}% change in reading speed "
            f"(95% CI: {d['pct_change_speed_ci_95'][0]:.1f}% to {d['pct_change_speed_ci_95'][1]:.1f}%)"
            if d['pct_change_speed_ci_95'] is not None else
            f"For readers with dyslexia (dyslexia_bin=1): Reader View combined coef (log-speed) = {d['coef_log_speed']:.4f}, "
            f"{sig_text(d['pvalue'])}."
        )
    else:
        interp_lines.append("No estimate available for dyslexic readers.")

    # Final concise answer to the yes/no question
    final_answer = "Cannot determine significance"  # default
    # Prefer dyslexic subgroup result to answer the question "Does Reader View improve reading speed for individuals with dyslexia?"
    if d is not None and d.get("pvalue") is not None:
        final_answer = ("Yes — Reader View improves reading speed for individuals with dyslexia (statistically significant)."
                        if d["pvalue"] < 0.05 and d["coef_log_speed"] > 0 else
                        "No — Reader View does not improve reading speed for individuals with dyslexia (not statistically significant or effect in opposite direction).")
    else:
        final_answer = "Insufficient information to answer whether Reader View improves reading speed for individuals with dyslexia."

    description = {
        "summary_lines": interp_lines,
        "final_answer": final_answer,
        "notes": out["notes"]
    }

    return {"object": out, "description": description}