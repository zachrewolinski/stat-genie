def extract_final_answer(model_output):
    """
    Extract key statistics about age effects and age-by-culture interactions
    from the provided modeling output dictionary.

    Returns a dict with:
      - "object": structured dict of extracted stats for each model and variable
      - "description": brief, data-driven interpretation of what those stats imply
    """
    import numpy as np
    results = {}
    description_lines = []

    # Helper to format a single stat entry
    def make_entry(coef, se, pval, ci_lower=None, ci_upper=None):
        entry = {
            "coef": float(coef) if coef is not None else None,
            "se": float(se) if se is not None else None,
            "p": float(pval) if pval is not None else None,
        }
        if ci_lower is not None and ci_upper is not None:
            entry["ci_lower"] = float(ci_lower)
            entry["ci_upper"] = float(ci_upper)
        return entry

    # Gather names of interaction variables we care about
    def varnames_from_result(res):
        # return list of variable names present in the result params
        try:
            params = res.params
            if hasattr(params, "index"):
                return list(params.index)
            # For MNLogit, params may be a DataFrame with index=variables and columns=equations
            try:
                return list(params.index)
            except Exception:
                return []
        except Exception:
            return []

    # Function to extract for GLM-like results (Series params)
    def extract_from_glm(res, var_list_prefixes):
        out = {}
        try:
            params = res.params  # pandas Series
            pvals = res.pvalues
            bse = getattr(res, "bse", None)
            ci = None
            try:
                ci = res.conf_int()
            except Exception:
                ci = None
            all_vars = list(params.index)
            # pick variables that match prefixes (prefix match)
            picked = [v for v in all_vars if any(v.startswith(pref) for pref in var_list_prefixes)]
            for v in picked:
                ci_lower, ci_upper = (None, None)
                if ci is not None and v in ci.index:
                    ci_lower, ci_upper = ci.loc[v].tolist()
                out[v] = make_entry(params.get(v, np.nan),
                                    bse.get(v, np.nan) if bse is not None else None,
                                    pvals.get(v, np.nan),
                                    ci_lower, ci_upper)
        except Exception as e:
            out["error"] = str(e)
        return out

    # 1) Multinomial: extract per-equation stats for age and interactions
    if "multinomial_model" in model_output and model_output["multinomial_model"] is not None:
        mn = model_output["multinomial_model"]
        mn_dict = {}
        try:
            params = mn.params  # DataFrame: index = variables, columns = equations
            pvals = mn.pvalues
            try:
                ci = mn.conf_int()
            except Exception:
                ci = None

            # determine variable list from index
            var_index = list(params.index)
            # find interactions present
            interaction_vars = [v for v in var_index if v.startswith("age_x_culture_")]
            # include age variables
            target_vars = ["age_centered", "age_centered_sq"] + interaction_vars

            # iterate over equations (columns). columns could be ints or strings
            for eq in params.columns:
                eq_label = str(eq)
                mn_dict[eq_label] = {}
                for v in target_vars:
                    if v in params.index:
                        coef = params.loc[v, eq]
                        # pvals may be a DataFrame shaped like params
                        pval = pvals.loc[v, eq] if (hasattr(pvals, "loc") and v in pvals.index) else None
                        se = None
                        try:
                            se = mn.bse.loc[v, eq]
                        except Exception:
                            # fallback approximate se
                            se = None
                        ci_lower = ci_upper = None
                        if ci is not None:
                            try:
                                # For MNLogit, conf_int may be multiindexed with column labels
                                if isinstance(ci, dict):
                                    # unlikely; skip
                                    pass
                                else:
                                    # ci might be a 3D structure; attempt to index
                                    ci_lower = ci.loc[(v, eq)][0] if (hasattr(ci, "index") and (v, eq) in ci.index) else None
                                    ci_upper = ci.loc[(v, eq)][1] if (hasattr(ci, "index") and (v, eq) in ci.index) else None
                            except Exception:
                                ci_lower = ci_upper = None
                        mn_dict[eq_label][v] = make_entry(coef, se, pval, ci_lower, ci_upper)
            results["multinomial"] = mn_dict

            # Summarize notable effects for description: look for significant (p<0.05) age and interactions
            sig_notes = []
            try:
                for eq in params.columns:
                    for v in target_vars:
                        try:
                            pval = float(pvals.loc[v, eq])
                            coef = float(params.loc[v, eq])
                        except Exception:
                            continue
                        if pval < 0.05:
                            sig_notes.append(f"(multinomial eq={eq}) {v} coef={coef:.3f} p={pval:.3f}")
            except Exception:
                pass

            if sig_notes:
                description_lines.append(
                    "Multinomial model: significant effects found: " + "; ".join(sig_notes)
                )
            else:
                description_lines.append("Multinomial model: no significant age or age-by-culture interaction effects at p<0.05.")
        except Exception as e:
            results["multinomial_error"] = str(e)
            description_lines.append("Multinomial model: could not extract parameters: " + str(e))
    else:
        description_lines.append("Multinomial model not available in model_output.")

    # 2) Social_choice GLM
    if "social_choice_model" in model_output and model_output["social_choice_model"] is not None:
        soc = model_output["social_choice_model"]
        try:
            # Extract age terms and age_x_culture_* interactions
            var_prefixes = ["age_centered", "age_centered_sq", "age_x_culture_"]
            soc_stats = extract_from_glm(soc, var_prefixes)
            results["social_choice"] = soc_stats

            # Interpret key effects
            # Check main age term
            main_age = soc_stats.get("age_centered")
            if main_age:
                p = main_age.get("p")
                coef = main_age.get("coef")
                if p is not None and p < 0.05:
                    description_lines.append(f"Social-choice model: main linear age effect positive and significant (coef={coef:.3f}, p={p:.3f}) — older children more likely to choose demonstrated options overall.")
                elif p is not None and p < 0.10:
                    description_lines.append(f"Social-choice model: main linear age effect positive but marginal (coef={coef:.3f}, p={p:.3f}) — trend toward older children choosing demonstrated options more.")
                else:
                    description_lines.append("Social-choice model: no robust main linear age effect (p>=0.10).")

            # Look for significant interactions
            sig_interactions = []
            for v, entry in soc_stats.items():
                if v.startswith("age_x_culture_"):
                    p = entry.get("p")
                    if p is not None and p < 0.05:
                        sig_interactions.append((v, entry.get("coef"), p))
            if sig_interactions:
                parts = []
                for v, coef, p in sig_interactions:
                    parts.append(f"{v} coef={coef:.3f} p={p:.3f}")
                description_lines.append("Social-choice model: significant age-by-culture interactions: " + "; ".join(parts) +
                                         " — these indicate that age-related change in reliance on social information differs in those sites compared to the reference site.")
            else:
                description_lines.append("Social-choice model: no age-by-culture interactions reached p<0.05.")
        except Exception as e:
            results["social_choice_error"] = str(e)
            description_lines.append("Social-choice model: extraction error: " + str(e))
    else:
        description_lines.append("Social-choice model not available in model_output.")

    # 3) Majority preference GLM among demonstrated choices
    if "majority_pref_model" in model_output and model_output["majority_pref_model"] is not None:
        maj = model_output["majority_pref_model"]
        try:
            var_prefixes = ["age_centered", "age_centered_sq", "age_x_culture_"]
            maj_stats = extract_from_glm(maj, var_prefixes)
            results["majority_pref"] = maj_stats

            # Interpret: quadratic age term and interactions
            quad = maj_stats.get("age_centered_sq")
            if quad:
                p = quad.get("p")
                coef = quad.get("coef")
                if p is not None and p < 0.05:
                    description_lines.append(f"Majority-preference model: significant quadratic age effect (age_centered_sq coef={coef:.3f}, p={p:.3f}) — preference for majority changes nonlinearly with age.")
                else:
                    description_lines.append("Majority-preference model: no significant quadratic age effect (p>=0.05).")
            else:
                description_lines.append("Majority-preference model: quadratic age term not present in results.")

            # main linear age
            lin = maj_stats.get("age_centered")
            if lin:
                p = lin.get("p")
                coef = lin.get("coef")
                if p is not None and p < 0.05:
                    description_lines.append(f"Majority-preference model: linear age effect significant (coef={coef:.3f}, p={p:.3f}).")
                elif p is not None and p < 0.10:
                    description_lines.append(f"Majority-preference model: linear age effect marginal (coef={coef:.3f}, p={p:.3f}).")
                else:
                    description_lines.append("Majority-preference model: no robust linear age effect (p>=0.10).")

            # interactions
            sig_interactions = []
            for v, entry in maj_stats.items():
                if v.startswith("age_x_culture_"):
                    p = entry.get("p")
                    if p is not None and p < 0.05:
                        sig_interactions.append((v, entry.get("coef"), p))
            if sig_interactions:
                parts = [f"{v} coef={coef:.3f} p={p:.3f}" for (v, coef, p) in sig_interactions]
                description_lines.append("Majority-preference model: significant age-by-culture interactions: " + "; ".join(parts))
            else:
                description_lines.append("Majority-preference model: no age-by-culture interactions reached p<0.05.")
        except Exception as e:
            results["majority_pref_error"] = str(e)
            description_lines.append("Majority-preference model: extraction error: " + str(e))
    else:
        description_lines.append("Majority-preference model not available in model_output.")

    # Combine description
    description = " ".join(description_lines)

    return {"object": results, "description": description}