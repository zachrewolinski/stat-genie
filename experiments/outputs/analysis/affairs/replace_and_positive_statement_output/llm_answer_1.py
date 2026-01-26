def extract_final_answer(model_output):
    """
    Extracts statistics related to the effect of 'children_binary' on extramarital affairs
    from the provided model_output dict (expected to contain 'logit_model', optional 'nb_model',
    and the precomputed marginal effect/predicted probabilities).

    Returns a dict with keys:
      - "object": a dictionary with extracted numeric results for the logistic model and (if present) the NB model
      - "description": a concise plain-language interpretation of the results and statistical significance
    """
    import numpy as np
    out = {"object": {}, "description": ""}

    # Check presence of logit model
    logit = model_output.get("logit_model", None)
    if logit is None:
        out["object"]["logit"] = None
        out["description"] = "No logistic model found in model_output."
        return out

    # Extract logistic model statistics for children_binary
    try:
        coef = float(logit.params["children_binary"])
    except Exception:
        # fallback: access by position if name not present
        try:
            idx = list(logit.params.index).index("children_binary")
            coef = float(logit.params.iloc[idx])
        except Exception:
            coef = None

    try:
        se = float(logit.bse["children_binary"])
    except Exception:
        try:
            idx = list(logit.bse.index).index("children_binary")
            se = float(logit.bse.iloc[idx])
        except Exception:
            se = None

    # z (or t) statistic and p-value
    z = None
    pvalue = None
    if coef is not None and se is not None and se != 0:
        z = coef / se
        # statsmodels exposes pvalues
        try:
            pvalue = float(logit.pvalues["children_binary"])
        except Exception:
            try:
                pvalue = float(logit.pvalues.iloc[list(logit.params.index).index("children_binary")])
            except Exception:
                pvalue = None

    # 95% CI
    try:
        ci = logit.conf_int().loc["children_binary"].tolist()
        ci = [float(ci[0]), float(ci[1])]
    except Exception:
        try:
            ci_arr = logit.conf_int().values
            idx = list(logit.params.index).index("children_binary")
            ci = [float(ci_arr[idx, 0]), float(ci_arr[idx, 1])]
        except Exception:
            ci = None

    # predicted probabilities and marginal effect (precomputed by model function)
    marginal_effect = model_output.get("marginal_effect_children_on_prob_any_affair", None)
    p_with_children = model_output.get("predicted_prob_with_children_at_means", None)
    p_without_children = model_output.get("predicted_prob_without_children_at_means", None)

    # sample size
    try:
        nobs = int(logit.nobs)
    except Exception:
        try:
            nobs = int(len(logit.model.endog))
        except Exception:
            nobs = None

    logit_result = {
        "coef_children_binary_logit": coef,
        "se_children_binary_logit": se,
        "z_or_t": z,
        "pvalue_children_binary_logit": pvalue,
        "ci95_children_binary_logit": ci,
        "marginal_effect_on_prob_any_affair": marginal_effect,
        "predicted_prob_with_children_at_means": p_with_children,
        "predicted_prob_without_children_at_means": p_without_children,
        "nobs": nobs,
    }
    out["object"]["logit"] = logit_result

    # Negative binomial (or fallback) results, if present
    nb = model_output.get("nb_model", None)
    if nb is None:
        out["object"]["nb"] = None
    else:
        # Many GLM result wrappers store params/pvalues/bse similarly
        try:
            nb_coef = float(nb.params["children_binary"])
        except Exception:
            try:
                idx = list(nb.params.index).index("children_binary")
                nb_coef = float(nb.params.iloc[idx])
            except Exception:
                nb_coef = None

        try:
            nb_se = float(nb.bse["children_binary"])
        except Exception:
            try:
                idx = list(nb.bse.index).index("children_binary")
                nb_se = float(nb.bse.iloc[idx])
            except Exception:
                nb_se = None

        nb_pvalue = None
        if nb_coef is not None:
            try:
                nb_pvalue = float(nb.pvalues["children_binary"])
            except Exception:
                try:
                    nb_pvalue = float(nb.pvalues.iloc[list(nb.params.index).index("children_binary")])
                except Exception:
                    nb_pvalue = None

        # confidence interval
        try:
            nb_ci = nb.conf_int().loc["children_binary"].tolist()
            nb_ci = [float(nb_ci[0]), float(nb_ci[1])]
        except Exception:
            try:
                ci_arr = nb.conf_int().values
                idx = list(nb.params.index).index("children_binary")
                nb_ci = [float(ci_arr[idx, 0]), float(ci_arr[idx, 1])]
            except Exception:
                nb_ci = None

        # exponentiated coef -> incidence rate ratio (IRR)
        irr = None
        irr_ci = None
        if nb_coef is not None:
            try:
                irr = float(np.exp(nb_coef))
                if nb_ci is not None:
                    irr_ci = [float(np.exp(nb_ci[0])), float(np.exp(nb_ci[1]))]
            except Exception:
                irr = None

        # sample size for NB model
        try:
            nb_nobs = int(nb.nobs)
        except Exception:
            try:
                nb_nobs = int(len(nb.model.endog))
            except Exception:
                nb_nobs = None

        nb_result = {
            "coef_children_binary_count": nb_coef,
            "se_children_binary_count": nb_se,
            "pvalue_children_binary_count": nb_pvalue,
            "ci95_children_binary_count": nb_ci,
            "incidence_rate_ratio_children_binary": irr,
            "irr_ci95": irr_ci,
            "nobs": nb_nobs,
        }
        out["object"]["nb"] = nb_result

    # Build concise description based on significance (alpha=0.05) using logit p-value for primary inference
    desc_lines = []
    # interpret direction and magnitude from marginal effect if available, else from coef
    if marginal_effect is not None:
        # marginal effect is difference p1 - p0 where p1 has children=1. Negative means children associated with lower probability.
        me = marginal_effect
        me_pct = me * 100 if me is not None else None
        desc_lines.append(
            f"Estimated effect: having children is associated with a change of {me_pct:+.2f} percentage points in the probability of any extramarital affair (predicted at covariate means)."
            if me_pct is not None
            else "Estimated marginal effect for children not available."
        )
    else:
        # fallback to logit coef (approximate marginal on log-odds)
        if coef is not None:
            desc_lines.append(f"Logit coefficient for children_binary = {coef:.4f} (negative => lower log-odds of any affair).")

    # significance decision
    sig_text = ""
    if pvalue is None:
        sig_text = "Could not determine statistical significance (p-value not available)."
    else:
        if pvalue < 0.05:
            sig_text = f"The association is statistically significant (two-sided p = {pvalue:.3g} < 0.05)."
        else:
            sig_text = f"The association is not statistically significant (two-sided p = {pvalue:.3g} >= 0.05)."
    desc_lines.append(sig_text)

    # add additional info about predicted probabilities
    if p_with_children is not None and p_without_children is not None:
        desc_lines.append(
            f"Predicted probability at mean covariates: with children = {p_with_children:.3f}, without children = {p_without_children:.3f}."
        )

    # add NB summary if present
    if out["object"]["nb"] is not None:
        nb_res = out["object"]["nb"]
        if nb_res["incidence_rate_ratio_children_binary"] is not None:
            irr = nb_res["incidence_rate_ratio_children_binary"]
            p_nb = nb_res["pvalue_children_binary_count"]
            irr_line = f"Among those reporting any affairs, children are associated with an incidence-rate ratio = {irr:.3f}"
            if nb_res["irr_ci95"] is not None:
                irr_line += f" (95% CI [{nb_res['irr_ci95'][0]:.3f}, {nb_res['irr_ci95'][1]:.3f}])"
            if p_nb is not None:
                irr_line += f", p = {p_nb:.3g}."
            else:
                irr_line += "."
            desc_lines.append(irr_line)
        else:
            desc_lines.append("Count-model (NB) results present but children effect could not be fully extracted.")

    # final concise conclusion
    # Primary question: "Does having children decrease (if at all) the engagement in extramarital affairs?"
    conclusion = ""
    if pvalue is None:
        conclusion = "Based on the available model output, an effect estimate is reported but statistical significance could not be determined."
    else:
        if pvalue < 0.05:
            # direction: use marginal_effect if available else coef sign
            direction = "decrease" if (marginal_effect is not None and marginal_effect < 0) or (marginal_effect is None and coef is not None and coef < 0) else "increase"
            conclusion = f"Answer: Yes — having children is associated with a small statistically significant {direction} in engagement in extramarital affairs (primary evidence from the logistic model)."
        else:
            conclusion = "Answer: No strong evidence that having children decreases engagement in extramarital affairs (effect is not statistically significant in the logistic model)."

    desc_lines.append(conclusion)
    out["description"] = " ".join(desc_lines)

    return out