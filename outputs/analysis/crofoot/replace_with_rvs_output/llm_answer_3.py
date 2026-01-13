def extract_final_answer(model_output):
    """
    Extracts coefficients, clustered SEs, z-stats, p-values, odds ratios and 95% CIs
    for the key predictors in the fitted ClusteredResults-like object returned
    by the modeling function.

    Returns a dictionary with keys:
      - "object": a pandas.DataFrame with numeric results for:
           rel_size_z, focal_central, rel_size_x_central
           and additionally the combined effect of rel_size when focal is central
           (rel_size_when_focal_central = rel_size_z + rel_size_x_central).
      - "description": a short human-readable interpretation of those results
           in the context of the question.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Helper to get cov matrix as DataFrame
    cov = getattr(model_output, "cov_params", None)
    if cov is None:
        raise ValueError("Model output does not provide clustered covariance matrix as cov_params.")

    # Convert cov to DataFrame if it's ndarray
    if isinstance(cov, np.ndarray):
        # try to get index/columns from params
        params_raw = getattr(model_output, "params", None)
        if params_raw is not None and hasattr(params_raw, "index"):
            idx = params_raw.index
        else:
            idx = [f"b{i}" for i in range(cov.shape[0])]
        cov = pd.DataFrame(cov, index=idx, columns=idx)
    elif isinstance(cov, pd.DataFrame):
        cov = cov.copy()
    else:
        # try to coerce
        cov = pd.DataFrame(np.asarray(cov))

    params = getattr(model_output, "params", None)
    if params is None:
        raise ValueError("Model output does not provide params.")

    # Ensure params is a Series
    if not isinstance(params, pd.Series):
        try:
            # If params has index attribute (like numpy structured), preserve it
            if hasattr(params, "index"):
                params = pd.Series(params, index=params.index)
            else:
                params = pd.Series(list(params))
        except Exception:
            raise ValueError("Could not interpret params from model output as a pandas Series.")

    # align cov and params indices if possible
    try:
        if list(cov.index) != list(params.index):
            cov = cov.reindex(index=params.index, columns=params.index)
    except Exception:
        # if reindexing fails, proceed with what we have
        pass

    # Clustered standard errors (sqrt diag of cov)
    se = pd.Series(np.sqrt(np.diag(cov.values)), index=params.index)

    # z-stats and p-values
    # Ensure z and pvals are pandas Series so .loc works below
    z_vals = (params / se).astype(float)
    if not isinstance(z_vals, pd.Series):
        z_vals = pd.Series(z_vals, index=params.index)
    else:
        z_vals = pd.Series(z_vals, index=params.index)

    pvals = 2 * (1 - norm.cdf(np.abs(z_vals.values)))
    pvals = pd.Series(pvals, index=params.index)

    # Clustered CIs from params +/- z*se (z=1.96 for 95%)
    z_alpha = norm.ppf(0.975)
    ci_lower = pd.Series((params - z_alpha * se).astype(float), index=params.index)
    ci_upper = pd.Series((params + z_alpha * se).astype(float), index=params.index)

    # Odds ratios and CIs
    or_vals = pd.Series(np.exp(params).astype(float), index=params.index)
    or_lower = pd.Series(np.exp(ci_lower).astype(float), index=params.index)
    or_upper = pd.Series(np.exp(ci_upper).astype(float), index=params.index)

    # Terms of interest
    terms = ['rel_size_z', 'focal_central', 'rel_size_x_central']
    rows = []
    for t in terms:
        if t in params.index:
            rows.append({
                'term': t,
                'coef': float(params.loc[t]),
                'se': float(se.loc[t]) if t in se.index else float(np.nan),
                'z': float(z_vals.loc[t]),
                'p': float(pvals.loc[t]),
                'OR': float(or_vals.loc[t]),
                'CI_lower_OR': float(or_lower.loc[t]),
                'CI_upper_OR': float(or_upper.loc[t])
            })
        else:
            rows.append({
                'term': t,
                'coef': np.nan,
                'se': np.nan,
                'z': np.nan,
                'p': np.nan,
                'OR': np.nan,
                'CI_lower_OR': np.nan,
                'CI_upper_OR': np.nan
            })

    results_df = pd.DataFrame(rows).set_index('term')

    # Compute combined effect of rel_size when focal is central:
    # effect = coef(rel_size_z) + coef(rel_size_x_central)
    comb_name = 'rel_size_when_focal_central'
    if all(t in params.index for t in ['rel_size_z', 'rel_size_x_central']):
        coef_a = params.loc['rel_size_z']
        coef_b = params.loc['rel_size_x_central']
        coef_sum = coef_a + coef_b
        # Var(a+b) = Var(a) + Var(b) + 2*Cov(a,b)
        try:
            var_a = cov.loc['rel_size_z', 'rel_size_z']
            var_b = cov.loc['rel_size_x_central', 'rel_size_x_central']
            cov_ab = cov.loc['rel_size_z', 'rel_size_x_central']
        except Exception:
            # If covariance lookup fails, fallback to NaN
            var_a = np.nan
            var_b = np.nan
            cov_ab = np.nan

        if np.isfinite(var_a) and np.isfinite(var_b) and np.isfinite(cov_ab):
            se_sum = np.sqrt(var_a + var_b + 2 * cov_ab)
            z_sum = coef_sum / se_sum
            p_sum = 2 * (1 - norm.cdf(np.abs(z_sum)))
            ci_low_sum = coef_sum - z_alpha * se_sum
            ci_up_sum = coef_sum + z_alpha * se_sum
            or_sum = float(np.exp(coef_sum))
            or_low_sum = float(np.exp(ci_low_sum))
            or_up_sum = float(np.exp(ci_up_sum))

            combined_row = {
                'term': comb_name,
                'coef': float(coef_sum),
                'se': float(se_sum),
                'z': float(z_sum),
                'p': float(p_sum),
                'OR': or_sum,
                'CI_lower_OR': or_low_sum,
                'CI_upper_OR': or_up_sum
            }
        else:
            combined_row = {
                'term': comb_name,
                'coef': float(coef_sum),
                'se': np.nan,
                'z': np.nan,
                'p': np.nan,
                'OR': float(np.exp(coef_sum)),
                'CI_lower_OR': np.nan,
                'CI_upper_OR': np.nan
            }
    else:
        combined_row = {
            'term': comb_name,
            'coef': np.nan,
            'se': np.nan,
            'z': np.nan,
            'p': np.nan,
            'OR': np.nan,
            'CI_lower_OR': np.nan,
            'CI_upper_OR': np.nan
        }

    results_df = pd.concat([results_df, pd.DataFrame([combined_row]).set_index('term')])

    # Short interpretation
    # significance threshold
    alpha = 0.05
    interp_lines = []
    # rel_size main effect (when focal_central == 0)
    if 'rel_size_z' in results_df.index and not pd.isna(results_df.loc['rel_size_z', 'coef']):
        coef = results_df.loc['rel_size_z', 'coef']
        p = results_df.loc['rel_size_z', 'p']
        orv = results_df.loc['rel_size_z', 'OR']
        cil = results_df.loc['rel_size_z', 'CI_lower_OR']
        ciu = results_df.loc['rel_size_z', 'CI_upper_OR']
        sig = (p < alpha) if pd.notna(p) else False
        interp_lines.append(
            f"Relative group size (rel_size_z) has coefficient {coef:.3f}, OR={orv:.3f} "
            f"(95% CI {cil:.3f}–{ciu:.3f}), p={p:.3f} -> {'statistically significant' if sig else 'not statistically significant'} "
            f"at alpha={alpha} for non-central contests (focal_central=0)."
        )
    # interaction
    if 'rel_size_x_central' in results_df.index and not pd.isna(results_df.loc['rel_size_x_central', 'coef']):
        coef = results_df.loc['rel_size_x_central', 'coef']
        p = results_df.loc['rel_size_x_central', 'p']
        orv = results_df.loc['rel_size_x_central', 'OR']
        cil = results_df.loc['rel_size_x_central', 'CI_lower_OR']
        ciu = results_df.loc['rel_size_x_central', 'CI_upper_OR']
        sig = (p < alpha) if pd.notna(p) else False
        interp_lines.append(
            f"Interaction (rel_size_x_central) has coefficient {coef:.3f}, OR={orv:.3f} "
            f"(95% CI {cil:.3f}–{ciu:.3f}), p={p:.3f} -> {'statistically significant' if sig else 'not statistically significant'}."
        )
        interp_lines.append(
            "Interpretation: the interaction modifies the effect of relative group size when the focal group is more central."
        )
    # combined effect when focal is central
    if comb_name in results_df.index and not pd.isna(results_df.loc[comb_name, 'coef']):
        coef = results_df.loc[comb_name, 'coef']
        p = results_df.loc[comb_name, 'p']
        orv = results_df.loc[comb_name, 'OR']
        cil = results_df.loc[comb_name, 'CI_lower_OR']
        ciu = results_df.loc[comb_name, 'CI_upper_OR']
        sig = (p < alpha) if pd.notna(p) else False
        interp_lines.append(
            f"When the focal group is more central (focal_central=1), the combined rel_size effect is coef {coef:.3f}, "
            f"OR={orv:.3f} (95% CI {cil:.3f}–{ciu:.3f}), p={p:.3f} -> "
            f"{'statistically significant' if sig else 'not statistically significant'}."
        )
    # focal_central main effect
    if 'focal_central' in results_df.index and not pd.isna(results_df.loc['focal_central', 'coef']):
        coef = results_df.loc['focal_central', 'coef']
        p = results_df.loc['focal_central', 'p']
        orv = results_df.loc['focal_central', 'OR']
        cil = results_df.loc['focal_central', 'CI_lower_OR']
        ciu = results_df.loc['focal_central', 'CI_upper_OR']
        sig = (p < alpha) if pd.notna(p) else False
        interp_lines.append(
            f"Focal centrality main effect (focal_central) has coef {coef:.3f}, OR={orv:.3f} "
            f"(95% CI {cil:.3f}–{ciu:.3f}), p={p:.3f} -> {'significant' if sig else 'not significant'}; "
            "this is the effect of centrality when rel_size_z = 0 (average relative size)."
        )

    description = " ".join(interp_lines) if interp_lines else "No relevant terms found in model output."

    return {
        "object": results_df,
        "description": description
    }