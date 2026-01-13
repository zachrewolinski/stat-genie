def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the question:
      Does 'Reader View' improve reading speed for individuals with dyslexia?
    Expects model_output to be a dict with keys 'dyslexic_model' and 'interaction_model'
    (the same structure returned by the modeling function).
    Returns a dict with:
      - "object": a dict with extracted numeric results (or None if unavailable)
      - "description": a brief interpretation of the results in context
    """
    import math
    res_obj = {"dyslexic_model": None, "interaction_model": None}
    desc_lines = []

    # Helper: safe p-value / t-critical calculation using scipy if available, fallback to normal approx
    try:
        from scipy import stats
        def t_pvalue(tval, df):
            return 2 * stats.t.sf(abs(tval), df)
        def t_crit(df, alpha=0.05):
            return stats.t.ppf(1 - alpha/2, df)
    except Exception:
        # fallback: normal approx
        def t_pvalue(tval, df):
            # approximate with normal
            import math
            from math import erf
            z = abs(tval)
            # two-sided p-value from normal distribution
            p = 2 * (0.5 * (1 - math.erf(z / math.sqrt(2))))
            return p
        def t_crit(df, alpha=0.05):
            return 1.959963984540054  # approx z for 95% CI

    # Validate input structure
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict; cannot extract results."
        }

    dys_model = model_output.get("dyslexic_model", None)
    int_model = model_output.get("interaction_model", None)

    # If both models are None, return early with explanation
    if dys_model is None and int_model is None:
        return {
            "object": None,
            "description": (
                "Both 'dyslexic_model' and 'interaction_model' are None. "
                "No estimates are available. Likely there were too few dyslexic observations "
                "and the interaction model was not fitted or returned. Cannot determine whether "
                "'Reader View' improves reading speed for individuals with dyslexia from these results."
            )
        }

    # Function to locate parameter names robustly
    def find_param_name(params_index, include_terms, exclude_terms=None):
        exclude_terms = exclude_terms or []
        for name in params_index:
            if all(term in name for term in include_terms) and not any(ex in name for ex in exclude_terms):
                return name
        return None

    # 1) Extract from dyslexic-only model if available
    if dys_model is not None:
        try:
            params = dys_model.params
            bse = dys_model.bse
            ci = dys_model.conf_int()
            n_obs = int(dys_model.nobs) if hasattr(dys_model, "nobs") else None
            df_resid = float(dys_model.df_resid) if hasattr(dys_model, "df_resid") else None

            # find ReaderView main effect name
            main_name = find_param_name(params.index, include_terms=["ReaderView"], exclude_terms=["Dyslexia"])
            if main_name is None:
                # attempt looser match
                main_name = next((n for n in params.index if "ReaderView" in n and "Dyslexia" not in n), None)

            if main_name is None:
                dys_entry = {
                    "available": False,
                    "reason": "No parameter name matching 'ReaderView' found in dyslexic model parameters."
                }
                desc_lines.append("Dyslexic-only model fitted but no 'ReaderView' parameter found.")
            else:
                coef = float(params[main_name])
                se = float(bse[main_name])
                # compute t and p
                tstat = coef / se if se != 0 else float("nan")
                pval = float(t_pvalue(tstat, df_resid)) if df_resid is not None else None
                # CI
                ci_low, ci_high = float(ci.loc[main_name, 0]), float(ci.loc[main_name, 1])
                significant = (pval is not None and pval < 0.05)
                dys_entry = {
                    "available": True,
                    "n_obs": n_obs,
                    "parameter_name": main_name,
                    "coef": coef,
                    "se": se,
                    "t": tstat,
                    "p": pval,
                    "ci_95_lower": ci_low,
                    "ci_95_upper": ci_high,
                    "significant_at_0.05": bool(significant)
                }
                desc_lines.append(
                    f"Dyslexic-only model: ReaderView coef = {coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p = "
                    f"{pval:.4f} ({'significant' if significant else 'not significant'})."
                )
        except Exception as e:
            dys_entry = {"available": False, "reason": f"Error extracting dyslexic_model results: {e}"}
            desc_lines.append("Error extracting dyslexic-only model results: " + str(e))
        res_obj["dyslexic_model"] = dys_entry

    # 2) Extract from interaction model if available
    if int_model is not None:
        try:
            params = int_model.params
            bse = int_model.bse
            ci = int_model.conf_int()
            cov = int_model.cov_params()
            n_obs = int(int_model.nobs) if hasattr(int_model, "nobs") else None
            df_resid = float(int_model.df_resid) if hasattr(int_model, "df_resid") else None

            # find main ReaderView term (excluding interaction)
            main_name = find_param_name(params.index, include_terms=["ReaderView"], exclude_terms=["Dyslexia"])
            if main_name is None:
                main_name = next((n for n in params.index if "ReaderView" in n and "Dyslexia" not in n), None)

            interaction_name = find_param_name(params.index, include_terms=["ReaderView", "Dyslexia"])
            if interaction_name is None:
                # sometimes interaction order reversed
                interaction_name = find_param_name(params.index, include_terms=["Dyslexia", "ReaderView"])

            # Also find Dyslexia main effect name (for clarity)
            dys_main_name = find_param_name(params.index, include_terms=["Dyslexia"], exclude_terms=["ReaderView"])
            if dys_main_name is None:
                dys_main_name = next((n for n in params.index if "Dyslexia" in n and "ReaderView" not in n), None)

            int_entry = {"available": True, "n_obs": n_obs}

            # Extract main ReaderView effect (this is effect when Dyslexia == 0)
            if main_name is not None:
                coef_main = float(params[main_name])
                se_main = float(bse[main_name])
                t_main = coef_main / se_main if se_main != 0 else float("nan")
                p_main = float(t_pvalue(t_main, df_resid)) if df_resid is not None else None
                ci_low_main, ci_high_main = float(ci.loc[main_name, 0]), float(ci.loc[main_name, 1])
                int_entry["ReaderView_when_Dyslexia_0"] = {
                    "parameter_name": main_name,
                    "coef": coef_main,
                    "se": se_main,
                    "t": t_main,
                    "p": p_main,
                    "ci_95_lower": ci_low_main,
                    "ci_95_upper": ci_high_main,
                    "significant_at_0.05": bool(p_main is not None and p_main < 0.05)
                }
                desc_lines.append(
                    f"Interaction model (Dyslexia=0 baseline): ReaderView coef = {coef_main:.3f}, "
                    f"95% CI [{ci_low_main:.3f}, {ci_high_main:.3f}], p = {p_main:.4f}."
                )
            else:
                int_entry["ReaderView_when_Dyslexia_0"] = {"available": False, "reason": "No ReaderView main term found."}
                desc_lines.append("Interaction model: no ReaderView main term found.")

            # Extract interaction term and compute ReaderView effect for Dyslexia == 1
            if interaction_name is not None and main_name is not None:
                coef_inter = float(params[interaction_name])
                se_inter = float(bse[interaction_name])
                t_inter = coef_inter / se_inter if se_inter != 0 else float("nan")
                p_inter = float(t_pvalue(t_inter, df_resid)) if df_resid is not None else None
                ci_low_inter, ci_high_inter = float(ci.loc[interaction_name, 0]), float(ci.loc[interaction_name, 1])

                # Combined effect for Dyslexia == 1: coef_main + coef_inter
                combined_coef = coef_main + coef_inter
                # variance of sum = var(main) + var(inter) + 2*cov(main,inter)
                var_main = cov.loc[main_name, main_name]
                var_inter = cov.loc[interaction_name, interaction_name]
                covar = cov.loc[main_name, interaction_name]
                var_sum = var_main + var_inter + 2 * covar
                se_sum = math.sqrt(var_sum) if var_sum >= 0 else float("nan")
                t_sum = combined_coef / se_sum if se_sum != 0 and not math.isnan(se_sum) else float("nan")
                p_sum = float(t_pvalue(t_sum, df_resid)) if df_resid is not None else None
                # CI using t_crit
                tcrit = t_crit(df_resid) if df_resid is not None else t_crit(None)
                ci_low_sum = combined_coef - tcrit * se_sum if not math.isnan(se_sum) else float("nan")
                ci_high_sum = combined_coef + tcrit * se_sum if not math.isnan(se_sum) else float("nan")

                int_entry["interaction_term"] = {
                    "parameter_name": interaction_name,
                    "coef": coef_inter,
                    "se": se_inter,
                    "t": t_inter,
                    "p": p_inter,
                    "ci_95_lower": ci_low_inter,
                    "ci_95_upper": ci_high_inter,
                    "significant_at_0.05": bool(p_inter is not None and p_inter < 0.05)
                }
                int_entry["ReaderView_when_Dyslexia_1"] = {
                    "combined_coef": combined_coef,
                    "se": se_sum,
                    "t": t_sum,
                    "p": p_sum,
                    "ci_95_lower": ci_low_sum,
                    "ci_95_upper": ci_high_sum,
                    "significant_at_0.05": bool(p_sum is not None and p_sum < 0.05)
                }

                desc_lines.append(
                    f"Interaction term ({interaction_name}): coef = {coef_inter:.3f}, 95% CI [{ci_low_inter:.3f}, {ci_high_inter:.3f}], p = {p_inter:.4f}."
                )
                desc_lines.append(
                    f"Combined ReaderView effect for Dyslexia=1: coef = {combined_coef:.3f}, 95% CI [{ci_low_sum:.3f}, {ci_high_sum:.3f}], p = {p_sum:.4f}."
                )
            else:
                # If no interaction or missing main term, still attempt to report what exists
                if interaction_name is None:
                    int_entry["interaction_term"] = {"available": False, "reason": "No interaction parameter found in interaction model."}
                    desc_lines.append("Interaction model: no ReaderView:Dyslexia interaction term found.")
                else:
                    int_entry["interaction_term"] = {"available": False, "reason": "Cannot compute combined effect due to missing main term."}

            # include Dyslexia main if present
            if dys_main_name is not None:
                int_entry["Dyslexia_main"] = {
                    "parameter_name": dys_main_name,
                    "coef": float(params[dys_main_name]),
                    "se": float(bse[dys_main_name]),
                    "ci_95_lower": float(ci.loc[dys_main_name, 0]),
                    "ci_95_upper": float(ci.loc[dys_main_name, 1])
                }

        except Exception as e:
            int_entry = {"available": False, "reason": f"Error extracting interaction_model results: {e}"}
            desc_lines.append("Error extracting interaction model results: " + str(e))

        res_obj["interaction_model"] = int_entry

    # Build a concise description
    description = " ".join(desc_lines) if desc_lines else "No detailed results could be extracted."
    return {"object": res_obj, "description": description}