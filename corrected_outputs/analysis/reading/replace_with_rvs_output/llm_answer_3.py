def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of Reader View from a fitted statsmodels MixedLMResultsWrapper.
    
    Returns a dictionary with:
      - "object": dict of numeric results (coefficients, SEs, z/t, p-values, 95% CIs, multiplicative effects on original speed)
      - "description": human-readable interpretation answering whether Reader View improves reading speed for people with dyslexia.
    
    The function handles two main model forms:
      - Model includes interaction reader_view:dyslexia_bin -> computes effect of reader_view for dyslexia=0 and dyslexia=1 (combined).
      - Model without interaction -> reports main effect of reader_view.
    """
    import numpy as np
    import pandas as pd
    from math import exp

    res = model_output  # statsmodels MixedLMResultsWrapper

    # Basic parameter info
    params = pd.Series(res.params)
    bse = pd.Series(res.bse)
    try:
        tvals = pd.Series(res.tvalues)
    except Exception:
        # fallback: compute t as coef / se
        tvals = params / bse
    try:
        pvals = pd.Series(res.pvalues)
    except Exception:
        # pvalues may not be available per-param for some fit methods; fallback to NaNs
        pvals = pd.Series({k: np.nan for k in params.index})

    ci_df = None
    try:
        ci_df = res.conf_int()  # DataFrame with 2 columns: lower, upper
        # Ensure indexing matches params
        ci_df.index = params.index
    except Exception:
        ci_df = pd.DataFrame({ 'lower': params - 1.96 * bse, 'upper': params + 1.96 * bse }, index=params.index)

    # helper to safe t_test for a contrast vector
    def contrast_test(vec):
        # vec should be array-like length == number of params
        import numpy as _np
        vec = _np.asarray(vec, dtype=float).reshape((len(params),))
        # Statsmodels expects 2D (k,) or (1,k) works as well
        try:
            ct = res.t_test(vec)
            # extract values robustly
            eff = float(np.squeeze(np.asarray(ct.effect)))
            sd = float(np.squeeze(np.asarray(ct.sd)))
            t = float(np.squeeze(np.asarray(ct.tvalue)))
            # pvalue may be array or attribute
            pv = None
            if hasattr(ct, 'pvalue'):
                pv = ct.pvalue
            elif hasattr(ct, 'pvalues'):
                pv = ct.pvalues
            else:
                try:
                    pv = float(ct.pvalue())
                except Exception:
                    pv = np.nan
            # pvalue may be array
            if isinstance(pv, (list, tuple, np.ndarray, pd.Series)):
                pv = float(np.asarray(pv).reshape(-1)[0])
            ci = None
            try:
                ci_arr = ct.conf_int()
                # conf_int may be array-like shape (1,2)
                ci = (float(ci_arr[0, 0]), float(ci_arr[0, 1]))
            except Exception:
                # fallback using eff +/- 1.96*sd
                ci = (eff - 1.96 * sd, eff + 1.96 * sd)
            return {"effect": eff, "se": sd, "t": t, "p": pv, "ci95": ci}
        except Exception as e:
            # If t_test fails, approximate using linear combination and covariance matrix
            try:
                cov = res.cov_params()
                eff = float(np.dot(vec, params.values))
                var = float(np.dot(vec, np.dot(cov.values, vec)))
                sd = np.sqrt(var) if var >= 0 else np.nan
                t = eff / sd if sd and sd != 0 else np.nan
                # two-sided p-value from normal approximation
                from scipy import stats
                p = 2 * float(stats.norm.sf(abs(t))) if not np.isnan(t) else np.nan
                ci = (eff - 1.96 * sd, eff + 1.96 * sd) if not np.isnan(sd) else (np.nan, np.nan)
                return {"effect": eff, "se": sd, "t": t, "p": p, "ci95": ci}
            except Exception:
                return {"effect": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "ci95": (np.nan, np.nan)}

    # Find parameter names
    param_names = list(params.index)

    # Find interaction param name if present (contains both substrings)
    interaction_name = None
    for name in param_names:
        if ('reader_view' in name) and ('dyslexia_bin' in name):
            interaction_name = name
            break

    # Find reader_view main param name (prefer exact 'reader_view')
    reader_name = None
    if 'reader_view' in param_names:
        reader_name = 'reader_view'
    else:
        # fallback: pick a param that contains reader_view but is not interaction
        for name in param_names:
            if 'reader_view' in name and name != interaction_name:
                reader_name = name
                break

    # Find dyslexia main param name if present
    dys_name = None
    for name in param_names:
        if 'dyslexia_bin' in name and name != interaction_name:
            dys_name = name
            break

    results_object = {"params": {}, "reader_view": None, "interaction": None, "for_non_dyslexia": None, "for_dyslexia": None}

    # Include all raw params (coef, se, t, p, ci)
    for name in param_names:
        results_object["params"][name] = {
            "coef": float(params[name]),
            "se": float(bse.get(name, np.nan)),
            "t_or_z": float(tvals.get(name, np.nan)),
            "p": float(pvals.get(name, np.nan)) if not pd.isna(pvals.get(name, np.nan)) else np.nan,
            "ci95": (float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])) if ci_df is not None else (np.nan, np.nan)
        }

    # If interaction exists, compute combined effects
    if reader_name is not None and interaction_name is not None:
        # effect for non-dyslexic: just reader_name
        vec_non = np.zeros(len(param_names))
        vec_non[param_names.index(reader_name)] = 1.0
        res_non = contrast_test(vec_non)

        # effect for dyslexic: reader + interaction
        vec_dys = np.zeros(len(param_names))
        vec_dys[param_names.index(reader_name)] = 1.0
        vec_dys[param_names.index(interaction_name)] = 1.0
        res_dys = contrast_test(vec_dys)

        # interaction individual stats
        inter_stats = {
            "coef": float(params[interaction_name]),
            "se": float(bse.get(interaction_name, np.nan)),
            "t_or_z": float(tvals.get(interaction_name, np.nan)),
            "p": float(pvals.get(interaction_name, np.nan)) if not pd.isna(pvals.get(interaction_name, np.nan)) else np.nan,
            "ci95": (float(ci_df.loc[interaction_name, 0]), float(ci_df.loc[interaction_name, 1])) if ci_df is not None else (np.nan, np.nan)
        }

        # convert log effects to multiplicative multipliers on original speed
        mult_non = (exp(res_non["effect"]), exp(res_non["ci95"][0]), exp(res_non["ci95"][1])) if not np.isnan(res_non["effect"]) else (np.nan, np.nan, np.nan)
        mult_dys = (exp(res_dys["effect"]), exp(res_dys["ci95"][0]), exp(res_dys["ci95"][1])) if not np.isnan(res_dys["effect"]) else (np.nan, np.nan, np.nan)

        results_object["reader_view"] = {
            "param_name": reader_name,
            "coef": res_non["effect"],
            "se": res_non["se"],
            "t": res_non["t"],
            "p": res_non["p"],
            "ci95": res_non["ci95"],
            "multiplier_on_speed": mult_non  # (exp(coef), exp(ci_low), exp(ci_high))
        }
        results_object["for_non_dyslexia"] = results_object["reader_view"]

        results_object["interaction"] = {
            "param_name": interaction_name,
            **inter_stats
        }

        results_object["for_dyslexia"] = {
            "coef": res_dys["effect"],
            "se": res_dys["se"],
            "t": res_dys["t"],
            "p": res_dys["p"],
            "ci95": res_dys["ci95"],
            "multiplier_on_speed": mult_dys
        }

        # Interpretation text
        desc_lines = []
        desc_lines.append("Model includes reader_view × dyslexia interaction.")
        # Non-dyslexia
        desc_lines.append(
            f"Estimated Reader View effect for readers WITHOUT dyslexia: log-change = {res_non['effect']:.4f}, "
            f"95% CI [{res_non['ci95'][0]:.4f}, {res_non['ci95'][1]:.4f}], p = {res_non['p']:.3g}. "
            f"This corresponds to a multiplier on raw speed = {mult_non[0]:.3f} "
            f"(95% CI [{mult_non[1]:.3f}, {mult_non[2]:.3f}])."
        )
        # Dyslexia
        desc_lines.append(
            f"Estimated Reader View effect for readers WITH dyslexia: log-change = {res_dys['effect']:.4f}, "
            f"95% CI [{res_dys['ci95'][0]:.4f}, {res_dys['ci95'][1]:.4f}], p = {res_dys['p']:.3g}. "
            f"Corresponding multiplier on raw speed = {mult_dys[0]:.3f} "
            f"(95% CI [{mult_dys[1]:.3f}, {mult_dys[2]:.3f}])."
        )
        # Interaction significance
        inter_p = inter_stats.get("p", np.nan)
        if not pd.isna(inter_p) and inter_p < 0.05:
            desc_lines.append(
                f"The interaction term (difference in Reader View effect between dyslexic and non-dyslexic readers) "
                f"is statistically significant (coef = {inter_stats['coef']:.4f}, p = {inter_p:.3g}), "
                "which indicates the Reader View effect differs by dyslexia status."
            )
        else:
            desc_lines.append(
                f"The interaction term is not statistically significant (coef = {inter_stats['coef']:.4f}, p = {inter_p:.3g}), "
                "so there is no strong evidence that the Reader View effect differs between dyslexic and non-dyslexic readers."
            )

        # Final direct answer to task question:
        # Determine whether Reader View improves reading speed for individuals with dyslexia:
        p_dys = res_dys["p"]
        eff_dys_mult = mult_dys[0]
        if (not pd.isna(p_dys)) and p_dys < 0.05 and eff_dys_mult > 1.0:
            desc_lines.append("Conclusion: Reader View appears to significantly IMPROVE reading speed for individuals with dyslexia (statistically significant increase).")
        elif (not pd.isna(p_dys)) and p_dys < 0.05 and eff_dys_mult < 1.0:
            desc_lines.append("Conclusion: Reader View appears to significantly DECREASE reading speed for individuals with dyslexia (statistically significant decrease).")
        else:
            desc_lines.append("Conclusion: There is NO statistically significant evidence that Reader View changes reading speed for individuals with dyslexia (two-sided p >= 0.05).")

        description = " ".join(desc_lines)

    else:
        # No interaction present: report main effect of reader_view
        if reader_name is None:
            # Unexpected: no reader_view param found
            description = "No reader_view parameter found in the model results; cannot extract effect."
            results_object["reader_view"] = None
        else:
            coef = float(params[reader_name])
            se_ = float(bse.get(reader_name, np.nan))
            t_ = float(tvals.get(reader_name, np.nan))
            p_ = float(pvals.get(reader_name, np.nan)) if not pd.isna(pvals.get(reader_name, np.nan)) else np.nan
            ci_ = (float(ci_df.loc[reader_name, 0]), float(ci_df.loc[reader_name, 1])) if ci_df is not None else (np.nan, np.nan)
            mult = (exp(coef), exp(ci_[0]), exp(ci_[1])) if not pd.isna(coef) else (np.nan, np.nan, np.nan)

            results_object["reader_view"] = {
                "param_name": reader_name,
                "coef": coef,
                "se": se_,
                "t": t_,
                "p": p_,
                "ci95": ci_,
                "multiplier_on_speed": mult
            }

            desc_lines = []
            desc_lines.append(
                f"Model does NOT include an interaction with dyslexia. Estimated Reader View main effect: "
                f"log-change = {coef:.4f}, 95% CI [{ci_[0]:.4f}, {ci_[1]:.4f}], p = {p_:.3g}."
            )
            desc_lines.append(
                f"This corresponds to a multiplier on raw speed = {mult[0]:.3f} (95% CI [{mult[1]:.3f}, {mult[2]:.3f}])."
            )

            if (not pd.isna(p_)) and p_ < 0.05 and mult[0] > 1.0:
                desc_lines.append("Conclusion: Reader View significantly IMPROVES reading speed (applies to all readers in this model).")
            elif (not pd.isna(p_)) and p_ < 0.05 and mult[0] < 1.0:
                desc_lines.append("Conclusion: Reader View significantly DECREASES reading speed (applies to all readers in this model).")
            else:
                desc_lines.append("Conclusion: There is NO statistically significant evidence that Reader View changes reading speed (two-sided p >= 0.05).")

            # If dyslexia covariate was in the model but no interaction, we can't say Reader View differentially benefits dyslexic readers.
            if dys_name is not None:
                desc_lines.append("Because there is no interaction term, the model does not provide evidence that the Reader View effect differs by dyslexia status.")
            description = " ".join(desc_lines)

    return {"object": results_object, "description": description}