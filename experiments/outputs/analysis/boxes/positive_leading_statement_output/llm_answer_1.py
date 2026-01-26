def extract_final_answer(model_output):
    """
    Extract key statistics about age effects and age-by-culture interactions
    from the two fitted models in model_output.

    Expects model_output to be a dict with keys:
      - 'social_model' : statsmodels fitted result (LogitResults or GLMResults)
      - 'majority_model' : statsmodels fitted result

    Returns a dict with:
      - "object": nested dict with extracted numeric results for each model
      - "description": short text interpretation highlighting significance of
                       age main effect and age-by-culture interaction
    """
    import numpy as np
    import re

    out = {"object": {}, "description": ""}

    def analyze_model(result, model_name):
        res = {}
        params = result.params
        pvals = result.pvalues
        bse = result.bse
        ci = result.conf_int()
        cov = result.cov_params()

        # Basic age main effect
        if "age_c" in params.index:
            age_coef = float(params["age_c"])
            age_se = float(bse["age_c"])
            age_p = float(pvals["age_c"])
            age_ci = tuple(ci.loc["age_c"].tolist())
        else:
            # No explicit age term
            age_coef = None
            age_se = None
            age_p = None
            age_ci = (None, None)

        res["age"] = {"coef": age_coef, "se": age_se, "pval": age_p, "ci_95": age_ci}

        # Find interaction terms: those containing both 'age_c' and 'culture_cat' (robust to ordering)
        inter_names = [name for name in params.index if ("age_c" in name and "culture_cat" in name)]
        # Also handle possibility of names like 'C(culture_cat)[T.X]:age_c' (contains both substrings)
        # Build dict of interaction coefficients keyed by culture level (extracted from param name)
        interactions = {}
        for name in inter_names:
            coef = float(params[name])
            pval = float(pvals[name])
            se_i = float(bse[name])
            # Try to extract culture level from the parameter name using [T.<level>] pattern
            m = re.search(r"\[T\.([^]]+)\]", name)
            if m:
                level = m.group(1)
            else:
                # fallback: try to extract text after 'C(culture_cat)' or after last ':' or '.' 
                # (not guaranteed but best-effort)
                # remove 'age_c' and common tokens
                tmp = name
                tmp = tmp.replace("age_c", "")
                tmp = tmp.replace("C(culture_cat)", "")
                tmp = tmp.replace(":", "")
                tmp = tmp.replace("[", "").replace("]", "").replace("T.", "")
                level = tmp.strip("_ ").strip()
                if level == "":
                    level = name  # ultimate fallback
            # slope for that culture = main age coef + interaction coef
            if age_coef is not None:
                slope = age_coef + coef
                # compute slope SE using covariance: var(a+b)=var(a)+var(b)+2cov(a,b)
                try:
                    cov_aa = cov.loc["age_c", "age_c"]
                    cov_bb = cov.loc[name, name]
                    cov_ab = cov.loc["age_c", name]
                    slope_se = float(np.sqrt(cov_aa + cov_bb + 2 * cov_ab))
                except Exception:
                    slope_se = None
            else:
                slope = None
                slope_se = None

            interactions[level] = {
                "param_name": name,
                "coef": coef,
                "se": se_i,
                "pval": pval,
                "slope_with_age_main": slope,
                "slope_se": slope_se,
            }

        res["interactions"] = interactions

        # Reference (baseline) culture slope for age is just age_coef (if present)
        if age_coef is not None:
            try:
                slope_ref_se = float(np.sqrt(cov.loc["age_c", "age_c"]))
            except Exception:
                slope_ref_se = None
            res["baseline_age_slope"] = {"slope": age_coef, "se": slope_ref_se}
        else:
            res["baseline_age_slope"] = {"slope": None, "se": None}

        # Joint (Wald) test that all age-by-culture interactions are zero
        if len(inter_names) > 0:
            # Build restriction matrix R where each row picks out one interaction parameter
            k = len(params)
            idx_map = {name: i for i, name in enumerate(params.index)}
            R = np.zeros((len(inter_names), k))
            for j, name in enumerate(inter_names):
                R[j, idx_map[name]] = 1.0
            try:
                wt = result.wald_test(R)
                # wt.statistic may be scalar or array
                try:
                    stat = float(np.squeeze(wt.statistic))
                except Exception:
                    stat = None
                try:
                    pval_wald = float(np.squeeze(wt.pvalue))
                except Exception:
                    pval_wald = None
                df_wald = wt.df_denom if hasattr(wt, "df_denom") else (len(inter_names))
            except Exception:
                stat = None
                pval_wald = None
                df_wald = len(inter_names)
        else:
            stat = None
            pval_wald = None
            df_wald = 0

        res["interaction_wald_test"] = {"chi2_or_F": stat, "pval": pval_wald, "df": df_wald, "n_interactions": len(inter_names)}

        # Prepare a brief model-level summary
        # Determine whether age is "significant" at alpha=0.05
        age_sig = (age_p is not None and age_p < 0.05)
        inter_sig = (pval_wald is not None and pval_wald < 0.05)

        res["summary_sentence"] = (
            f"{model_name}: age main effect coef={age_coef:.4f}" if age_coef is not None else f"{model_name}: no age main term"
        )
        if age_coef is not None:
            res["summary_sentence"] += f" (p={age_p:.3g}); "
        if len(interactions) > 0:
            res["summary_sentence"] += f"{len(interactions)} age-by-culture interaction(s) found; joint-test p={pval_wald:.3g}"
        else:
            res["summary_sentence"] += " no age-by-culture interaction terms present."

        return res

    # Process social_model
    if "social_model" in model_output and model_output["social_model"] is not None:
        try:
            social_res = analyze_model(model_output["social_model"], "SocialInfoUsed_model")
            out["object"]["social_model"] = social_res
        except Exception as e:
            out["object"]["social_model_error"] = str(e)
    else:
        out["object"]["social_model_error"] = "social_model missing"

    # Process majority_model
    if "majority_model" in model_output and model_output["majority_model"] is not None:
        try:
            majority_res = analyze_model(model_output["majority_model"], "MajorityChoice_model")
            out["object"]["majority_model"] = majority_res
        except Exception as e:
            out["object"]["majority_model_error"] = str(e)
    else:
        out["object"]["majority_model_error"] = "majority_model missing"

    # Build a concise description that answers the yes/no question about whether
    # reliance on social information and preference for majority cues change with age
    desc_parts = []
    sm = out["object"].get("social_model")
    if sm and "age" in sm and sm["age"]["pval"] is not None:
        age_p = sm["age"]["pval"]
        if age_p < 0.05:
            desc_parts.append("Reliance on social information changes with age (social_model: age p < 0.05).")
        else:
            desc_parts.append("No strong evidence that reliance on social information changes with age (social_model: age p >= 0.05).")
        # Interaction
        ip = sm["interaction_wald_test"].get("pval")
        if ip is not None:
            if ip < 0.05:
                desc_parts.append("The age-related change differs across cultures (significant age×culture interaction in social_model).")
            else:
                desc_parts.append("Age-related change does not differ strongly across cultures (non-significant age×culture interaction in social_model).")
    else:
        desc_parts.append("Could not assess age effect in social_model from available output.")

    mm = out["object"].get("majority_model")
    if mm and "age" in mm and mm["age"]["pval"] is not None:
        age_p = mm["age"]["pval"]
        if age_p < 0.05:
            desc_parts.append("Preference for majority cues among those who used social information changes with age (majority_model: age p < 0.05).")
        else:
            desc_parts.append("No strong evidence that preference for majority cues changes with age (majority_model: age p >= 0.05).")
        ip = mm["interaction_wald_test"].get("pval")
        if ip is not None:
            if ip < 0.05:
                desc_parts.append("The developmental trajectory of majority preference differs across cultures (significant age×culture interaction in majority_model).")
            else:
                desc_parts.append("The developmental trajectory of majority preference does not differ strongly across cultures (non-significant age×culture interaction in majority_model).")
    else:
        desc_parts.append("Could not assess age effect in majority_model from available output.")

    out["description"] = " ".join(desc_parts)

    return out