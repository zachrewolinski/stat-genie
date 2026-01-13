def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of age on choosing the majority option
    from the provided GEE model output.

    Expects model_output to be a dict with keys 'main' and 'interaction' containing
    statsmodels GEEResultsWrapper objects (as produced by the model code).
    
    Returns:
      {
        "object": <dict with extracted numeric results>,
        "description": <text interpreting those results in the study context>
      }
    """
    out = {
        "main_age": None,
        "interaction_age_terms": None,
        "interaction_joint_test": None
    }

    try:
        res_main = model_output.get('main', None)
        res_int = model_output.get('interaction', None)
    except Exception as e:
        return {
            "object": None,
            "description": f"Error accessing model_output: {e}"
        }

    # Helper to safely get attributes
    def safe_get(res, attr, default=None):
        try:
            return getattr(res, attr)
        except Exception:
            return default

    # Extract main effect of age from main model
    if res_main is None:
        main_age_info = None
    else:
        try:
            params = safe_get(res_main, "params")
            pvalues = safe_get(res_main, "pvalues")
            bse = safe_get(res_main, "bse")
            conf = safe_get(res_main, "conf_int")() if callable(safe_get(res_main, "conf_int")) else safe_get(res_main, "conf_int")
            # conf_int may be a method or attribute; handle both
            if hasattr(res_main, "conf_int") and not callable(res_main.conf_int):
                conf = res_main.conf_int
        except Exception:
            params = getattr(res_main, "params", None)
            pvalues = getattr(res_main, "pvalues", None)
            bse = getattr(res_main, "bse", None)
            conf = None

        if params is None or 'age_years' not in params.index:
            main_age_info = None
        else:
            coef = float(params.loc['age_years'])
            se = float(bse.loc['age_years']) if (bse is not None and 'age_years' in bse.index) else None
            pval = float(pvalues.loc['age_years']) if (pvalues is not None and 'age_years' in pvalues.index) else None
            ci = None
            try:
                if conf is not None:
                    # conf may be DataFrame-like
                    ci_row = conf.loc['age_years']
                    ci = [float(ci_row.iloc[0]), float(ci_row.iloc[1])]
            except Exception:
                ci = None

            # z-statistic if SE available
            z = float(coef / se) if (se is not None and se != 0) else None

            main_age_info = {
                "coef_log_odds": coef,
                "se": se,
                "z": z,
                "p_value": pval,
                "95%CI_log_odds": ci,
                "interpretation": (
                    "Positive coef => greater log-odds of choosing the majority with increasing age; "
                    "negative => lower log-odds with increasing age."
                )
            }

    out["main_age"] = main_age_info

    # Extract interaction terms from interaction model
    interaction_info = {}
    joint_test_info = None
    if res_int is None:
        interaction_info = None
    else:
        try:
            params_int = res_int.params
            pvalues_int = res_int.pvalues
            bse_int = res_int.bse
            conf_int_df = res_int.conf_int()
        except Exception:
            params_int = getattr(res_int, "params", None)
            pvalues_int = getattr(res_int, "pvalues", None)
            bse_int = getattr(res_int, "bse", None)
            conf_int_df = None

        # Identify interaction terms involving age (exclude the main 'age_years' term)
        interaction_names = []
        if params_int is not None:
            for name in params_int.index:
                # Common naming from patsy/statsmodels: 'age_years:C(site_id)[T.<level>]' or 'age_years:C(site_id)[T.x]'
                # Also sometimes 'age_years: C(site_id)[T.x]' (with space). Use substring 'age_years' and ':' to find interactions.
                if name != 'age_years' and 'age_years' in name:
                    # ensure this is an interaction (contains ':' or 'C(site_id)' part)
                    if (':' in name) or ('C(site_id)' in name):
                        interaction_names.append(name)

        # Build dict of stats for each interaction term found
        terms_dict = {}
        for name in interaction_names:
            try:
                coef = float(params_int.loc[name])
            except Exception:
                coef = None
            try:
                se = float(bse_int.loc[name]) if (bse_int is not None and name in bse_int.index) else None
            except Exception:
                se = None
            try:
                pval = float(pvalues_int.loc[name]) if (pvalues_int is not None and name in pvalues_int.index) else None
            except Exception:
                pval = None
            try:
                ci_row = conf_int_df.loc[name] if (conf_int_df is not None and name in conf_int_df.index) else None
                ci = [float(ci_row.iloc[0]), float(ci_row.iloc[1])] if ci_row is not None else None
            except Exception:
                ci = None

            terms_dict[name] = {
                "coef_log_odds": coef,
                "se": se,
                "p_value": pval,
                "95%CI_log_odds": ci,
                "interpretation": (
                    "Interaction term: how the age slope differs for this site relative to the reference site. "
                    "Positive coef => age effect is larger (more positive) than reference site."
                )
            }

        interaction_info = terms_dict

        # Joint Wald test for all interaction terms = 0 (i.e., test whether age slope differs across sites)
        if len(interaction_names) > 0:
            # Build constraint string "name1 = 0, name2 = 0, ..."
            constraint = ", ".join([f"{n} = 0" for n in interaction_names])
            try:
                wres = res_int.wald_test(constraint)
                # wres may have attributes statistic and pvalue (or .summary())
                stat = None
                pval = None
                df_denom = None
                try:
                    stat = float(wres.statistic) if hasattr(wres, "statistic") else None
                except Exception:
                    # sometimes statistic is array-like
                    try:
                        stat = float(wres.statistic[0][0])
                    except Exception:
                        stat = None
                try:
                    pval = float(wres.pvalue) if hasattr(wres, "pvalue") else None
                except Exception:
                    try:
                        pval = float(wres.pvalues) if hasattr(wres, "pvalues") else None
                    except Exception:
                        pval = None
                joint_test_info = {
                    "constraint": constraint,
                    "wald_statistic": stat,
                    "p_value": pval,
                    "conclusion": (
                        "If p_value < 0.05, there is evidence that the age slope differs across sites (interaction present). "
                        "If p_value >= 0.05, no evidence that age slope differs across sites."
                    )
                }
            except Exception:
                # If wald_test fails for some reason, fall back to checking if any interaction term p < .05
                any_sig = any((terms_dict[n]["p_value"] is not None and terms_dict[n]["p_value"] < 0.05) for n in terms_dict)
                joint_test_info = {
                    "constraint": constraint,
                    "wald_test_failed": True,
                    "fallback_any_interaction_p_lt_0.05": any_sig,
                    "note": "Could not compute joint Wald test; fallback indicates whether any single interaction term is individually significant at p<0.05."
                }
        else:
            joint_test_info = {
                "constraint": None,
                "wald_statistic": None,
                "p_value": None,
                "conclusion": "No interaction terms involving age were found in the interaction model."
            }

    out["interaction_age_terms"] = interaction_info
    out["interaction_joint_test"] = joint_test_info

    # Prepare short human-readable description/interpretation
    desc_lines = []
    if out["main_age"] is None:
        desc_lines.append("Main model: could not extract age effect.")
    else:
        coef = out["main_age"]["coef_log_odds"]
        p = out["main_age"]["p_value"]
        ci = out["main_age"]["95%CI_log_odds"]
        if coef is not None:
            direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
            desc_lines.append(
                f"Main model: age (per year) has coef (log-odds) = {coef:.4f}" +
                (f", 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "") +
                (f", p = {p:.4g}" if p is not None else "")
                + f". This indicates a {direction} in the likelihood of choosing the majority with age."
            )
        else:
            desc_lines.append("Main model: age effect present but numeric details unavailable.")

    # Interaction interpretation
    if out["interaction_age_terms"] is None:
        desc_lines.append("Interaction model: could not extract interaction information.")
    else:
        if isinstance(out["interaction_age_terms"], dict) and len(out["interaction_age_terms"]) > 0:
            # Summarize whether any interaction terms are individually significant
            sig_terms = [n for n, v in out["interaction_age_terms"].items() if v.get("p_value") is not None and v["p_value"] < 0.05]
            if out["interaction_joint_test"] is not None and out["interaction_joint_test"].get("p_value") is not None:
                jp = out["interaction_joint_test"]["p_value"]
                if jp < 0.05:
                    desc_lines.append("Interaction model: joint test indicates that the age slope differs across sites (joint p < 0.05).")
                else:
                    desc_lines.append("Interaction model: joint test indicates no evidence that the age slope differs across sites (joint p >= 0.05).")
            else:
                # fall back to individual terms
                if len(sig_terms) > 0:
                    desc_lines.append(
                        f"Interaction model: some site-specific age interactions are individually significant (examples: {sig_terms[:3]}...). "
                        "This suggests developmental trajectories may differ for some sites."
                    )
                else:
                    desc_lines.append("Interaction model: no individual age-by-site interaction terms are individually significant at p<0.05.")

        else:
            desc_lines.append("Interaction model: no age-by-site interaction terms found.")

    description = " ".join(desc_lines)

    return {
        "object": out,
        "description": description
    }