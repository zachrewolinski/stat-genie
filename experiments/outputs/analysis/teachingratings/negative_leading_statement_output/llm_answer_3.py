def extract_final_answer(model_output):
    """
    Extracts statistics for the 'beauty_z' coefficient from the two models returned by the
    modeling function and returns a concise summary plus the numeric results.

    Expects model_output to be a dict with keys:
      - 'model1_clustered_by_prof'
      - 'model2_prof_fixed_effects_clustered_by_prof'

    Returns:
      {
        "object": { <model_key>: {coef, se, t, p, ci_lower, ci_upper, eval_sd,
                                  effect_SD_of_eval, percent_of_1-5_scale,
                                  significant_p_lt_0.05}, ... },
        "description": "<human-readable summary of findings>"
      }
    """
    import numpy as np

    results_summary = {}

    # Accept either the exact expected keys or the first two RegressionResults in the dict
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing the fitted model result objects.")

    # Define which keys to look for (fall back to whatever keys are present)
    preferred_keys = ['model1_clustered_by_prof', 'model2_prof_fixed_effects_clustered_by_prof']
    keys = [k for k in preferred_keys if k in model_output]
    if len(keys) < 2:
        # fallback: try to use the first two items in the dict
        keys = list(model_output.keys())[:2]

    for key in keys:
        res_obj = model_output.get(key)
        if res_obj is None:
            continue

        try:
            params = getattr(res_obj, 'params')
            bse = getattr(res_obj, 'bse')
            tvals = getattr(res_obj, 'tvalues')
            pvals = getattr(res_obj, 'pvalues')
            conf = getattr(res_obj, 'conf_int')()

            # Get index/position of beauty_z
            if hasattr(params, 'get') and 'beauty_z' in params.index:
                coef = float(params['beauty_z'])
                se = float(bse['beauty_z'])
                t = float(tvals['beauty_z'])
                p = float(pvals['beauty_z'])
                # confidence interval extraction handling DataFrame/ndarray
                try:
                    ci_row = conf.loc['beauty_z']
                    ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
                except Exception:
                    # conf might be ndarray; find index by parameter order
                    idx = list(params.index).index('beauty_z')
                    ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])
            else:
                raise KeyError("Coefficient 'beauty_z' not found in model parameters for key '{}'.".format(key))

            # Compute approximate SD of the dependent variable used in the model (if available)
            endog = getattr(res_obj.model, 'endog', None)
            if endog is not None:
                try:
                    eval_sd = float(np.std(endog, ddof=1))
                except Exception:
                    eval_sd = None
            else:
                eval_sd = None

            effect_sd = (coef / eval_sd) if (eval_sd is not None and eval_sd != 0) else None

            # Express coef as percent of the 1-5 scale's range (4 points)
            percent_of_scale = (coef / 4.0) * 100.0

            results_summary[key] = {
                'coef': coef,
                'se': se,
                't': t,
                'p': p,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'eval_sd': eval_sd,
                'effect_SD_of_eval': effect_sd,
                'percent_of_1-5_scale': percent_of_scale,
                'significant_p_lt_0.05': (p < 0.05)
            }
        except Exception as e:
            results_summary[key] = {'error': str(e)}

    # Build a brief interpretation based on extracted stats
    sig_models = [k for k, v in results_summary.items() if isinstance(v, dict) and v.get('significant_p_lt_0.05') is True]
    non_sig_models = [k for k, v in results_summary.items() if isinstance(v, dict) and v.get('significant_p_lt_0.05') is False]

    desc_lines = []
    desc_lines.append("Extracted statistics for the coefficient on 'beauty_z' (one SD increase in perceived beauty):")
    for k, v in results_summary.items():
        if 'error' in v:
            desc_lines.append(f"- {k}: error extracting stats: {v['error']}")
            continue
        sign = "positive" if v['coef'] > 0 else ("negative" if v['coef'] < 0 else "zero")
        sig_text = "statistically significant (p < 0.05)" if v['significant_p_lt_0.05'] else "not statistically significant (p >= 0.05)"
        # Short numeric summary
        desc_lines.append(
            "- {k}: coef = {coef:.4f}, 95% CI [{lo:.4f}, {hi:.4f}], se = {se:.4f}, p = {p:.3g}; {sign}, {sig_text}."
            .format(k=k, coef=v['coef'], lo=v['ci_lower'], hi=v['ci_upper'], se=v['se'], p=v['p'], sign=sign, sig_text=sig_text)
        )
        if v.get('eval_sd') is not None:
            desc_lines.append(
                "    -> This corresponds to about {pct:.2f}% of the 1-5 scale range and ≈{es:.3f} SDs of eval."
                .format(pct=v['percent_of_1-5_scale'], es=(v['effect_SD_of_eval'] if v['effect_SD_of_eval'] is not None else float('nan')))
            )

    # Overall conclusion
    if len(sig_models) == 0:
        desc_lines.append("Overall conclusion: No evidence of a statistically significant effect of instructor beauty on student evaluations at alpha = 0.05 in either specification.")
    else:
        desc_lines.append("Overall conclusion: The coefficient on beauty is statistically significant in the following model(s): " + ", ".join(sig_models) + ". Interpret the sign and magnitude from the reported coefficients above.")

    description = " ".join(desc_lines)

    return {
        "object": results_summary,
        "description": description
    }