def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of instructor beauty on student evaluations
    from the model_output returned by the modeling function.

    Returns:
      {
        "object": {
          "ols": { ... },         # coefficients, SEs, t-stats, p-values, 95% CIs,
                                 # marginal effects for male & female instructors
          "mixedlm": { ... } OR None  # fixed-effect beauty coefficient summary from mixed model
        },
        "description": "..."      # brief interpretation in context
      }
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    out = {"object": {}, "description": ""}

    # Get models
    ols = model_output.get('ols', None)
    mixed = model_output.get('mixedlm', None)

    if ols is None:
        raise ValueError("OLS result missing from model_output['ols'].")

    # --- OLS (with beauty * female interaction) ---
    try:
        params = ols.params  # pandas Series
        cov = ols.cov_params()  # covariance matrix used for clustered SEs
        df_resid = float(ols.df_resid)

        # Required parameter names
        b_name = 'beauty_c'
        int_name = 'beauty_c:female'  # name produced by patsy/statsmodels for interaction

        beta_b = float(params.get(b_name, np.nan))
        beta_int = float(params.get(int_name, 0.0))  # 0 if not present

        # Standard errors for components
        se_b = float(np.sqrt(cov.loc[b_name, b_name])) if (b_name in cov.index) else np.nan
        se_int = float(np.sqrt(cov.loc[int_name, int_name])) if (int_name in cov.index) else np.nan
        cov_bi = float(cov.loc[b_name, int_name]) if (b_name in cov.index and int_name in cov.index) else 0.0

        # Marginal effects: male (female=0) and female (female=1)
        eff_male = beta_b
        se_male = se_b
        t_male = eff_male / se_male if se_male != 0 and not np.isnan(se_male) else np.nan
        p_male = 2.0 * stats.t.sf(abs(t_male), df_resid) if not np.isnan(t_male) else np.nan
        tcrit = stats.t.ppf(1 - 0.025, df_resid)
        ci_male = (eff_male - tcrit * se_male, eff_male + tcrit * se_male) if not np.isnan(se_male) else (np.nan, np.nan)

        eff_female = beta_b + beta_int
        # Var(effect_female) = Var(beta_b) + Var(beta_int) + 2 Cov(beta_b, beta_int)
        var_female = 0.0
        if (b_name in cov.index) and (int_name in cov.index):
            var_female = cov.loc[b_name, b_name] + cov.loc[int_name, int_name] + 2.0 * cov.loc[b_name, int_name]
            se_female = float(np.sqrt(max(var_female, 0.0)))
        else:
            se_female = np.nan

        t_female = eff_female / se_female if se_female != 0 and not np.isnan(se_female) else np.nan
        p_female = 2.0 * stats.t.sf(abs(t_female), df_resid) if not np.isnan(t_female) else np.nan
        ci_female = (eff_female - tcrit * se_female, eff_female + tcrit * se_female) if not np.isnan(se_female) else (np.nan, np.nan)

        # Also report the interaction coefficient itself
        t_int = (beta_int / se_int) if (se_int != 0 and not np.isnan(se_int)) else np.nan
        p_int = 2.0 * stats.t.sf(abs(t_int), df_resid) if not np.isnan(t_int) else np.nan
        ci_int = (beta_int - tcrit * se_int, beta_int + tcrit * se_int) if not np.isnan(se_int) else (np.nan, np.nan)

        ols_summary = {
            "coef_beauty": float(beta_b),
            "se_beauty": float(se_b),
            "t_beauty": float(ols.tvalues[b_name]) if b_name in ols.tvalues.index else t_male,
            "p_beauty": float(ols.pvalues[b_name]) if b_name in ols.pvalues.index else p_male,
            "ci95_beauty": tuple(map(float, ols.conf_int().loc[b_name])) if b_name in ols.conf_int().index else ci_male,
            "coef_interaction_beauty_female": float(beta_int),
            "se_interaction": float(se_int) if not np.isnan(se_int) else None,
            "t_interaction": float(ols.tvalues[int_name]) if int_name in ols.tvalues.index else t_int,
            "p_interaction": float(ols.pvalues[int_name]) if int_name in ols.pvalues.index else p_int,
            "ci95_interaction": tuple(map(float, ols.conf_int().loc[int_name])) if int_name in ols.conf_int().index else ci_int,
            "marginal_effect_male": {
                "effect": float(eff_male),
                "se": float(se_male),
                "t": float(t_male),
                "p": float(p_male),
                "ci95": (float(ci_male[0]), float(ci_male[1]))
            },
            "marginal_effect_female": {
                "effect": float(eff_female),
                "se": float(se_female) if not np.isnan(se_female) else None,
                "t": float(t_female) if not np.isnan(t_female) else None,
                "p": float(p_female) if not np.isnan(p_female) else None,
                "ci95": (float(ci_female[0]), float(ci_female[1])) if not np.isnan(se_female) else (None, None)
            },
            "notes": "OLS used cluster-robust SEs clustered on professor id; df for tests = residual df."
        }

        out["object"]["ols"] = ols_summary

    except Exception as e:
        out["object"]["ols"] = None
        out["description"] += f"Failed to extract OLS results: {e}\n"

    # --- Mixed model (random intercept by professor) ---
    if mixed is None:
        out["object"]["mixedlm"] = None
    else:
        try:
            mparams = mixed.params  # pandas Series
            mcov = mixed.cov_params()  # covariance of fixed-effects estimates
            # For mixedlm p-values, statsmodels typically provides mixed.pvalues; we may use z-approx
            from scipy.stats import norm

            b_name = 'beauty_c'
            if b_name in mparams.index:
                beta_m = float(mparams[b_name])
                se_m = float(np.sqrt(mcov.loc[b_name, b_name]))
                z_val = beta_m / se_m if se_m != 0 else np.nan
                p_z = 2.0 * norm.sf(abs(z_val)) if not np.isnan(z_val) else np.nan
                ci_low = beta_m - 1.96 * se_m
                ci_high = beta_m + 1.96 * se_m

                mixed_summary = {
                    "coef_beauty": beta_m,
                    "se_beauty": se_m,
                    "z_beauty": float(z_val) if not np.isnan(z_val) else None,
                    "p_beauty_z_approx": float(p_z) if not np.isnan(p_z) else None,
                    "ci95_beauty_normal_approx": (float(ci_low), float(ci_high)),
                    "notes": "Mixed model used random intercept by professor; p-value is z-approx (normal)."
                }
            else:
                mixed_summary = {"error": "beauty_c not found in mixed model fixed effects."}

            out["object"]["mixedlm"] = mixed_summary

        except Exception as e:
            out["object"]["mixedlm"] = None
            out["description"] += f"Failed to extract mixedlm results: {e}\n"

    # --- Construct brief interpretation/description ---
    # Prefer OLS marginal effects for interpretation (gives male & female differences).
    try:
        ols_obj = out["object"]["ols"]
        if ols_obj is not None:
            eff_male = ols_obj["marginal_effect_male"]["effect"]
            p_male = ols_obj["marginal_effect_male"]["p"]
            eff_female = ols_obj["marginal_effect_female"]["effect"]
            p_female = ols_obj["marginal_effect_female"]["p"]
            int_coef = ols_obj["coef_interaction_beauty_female"]
            p_int = ols_obj["p_interaction"] if "p_interaction" in ols_obj else None

            desc_lines = []
            desc_lines.append(
                "Interpretation (OLS, clustered SEs): "
                f"For male instructors, a one-unit increase in (mean-centered) beauty is associated with a change of {eff_male:.3f} points in the course evaluation "
                f"(p = {p_male:.3g})."
            )
            desc_lines.append(
                f"For female instructors, the marginal effect is {eff_female:.3f} points (p = {p_female:.3g}). "
                "This female marginal effect equals beauty coefficient + interaction coefficient."
            )
            desc_lines.append(
                f"The beauty × female interaction coefficient is {int_coef:.3f} (p ≈ {p_int:.3g}), "
                "so if this interaction is statistically significant it indicates the beauty effect differs by instructor gender."
            )
            if out["object"]["mixedlm"] is not None:
                m = out["object"]["mixedlm"]
                desc_lines.append(
                    "Mixed-effects model (random intercept by professor) gives a fixed-effect beauty coefficient of "
                    f"{m['coef_beauty']:.3f} (se = {m['se_beauty']:.3f}, p_z ≈ {m['p_beauty_z_approx']:.3g})."
                )
            out["description"] = " ".join(desc_lines)
    except Exception:
        # fallback short description
        out["description"] = "Extracted coefficients, SEs, CIs, and p-values for beauty effect (and beauty×female interaction) from OLS and mixed models."

    return out