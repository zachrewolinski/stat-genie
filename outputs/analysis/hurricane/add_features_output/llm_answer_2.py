def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, test statistics, p-values, and 95% CIs
    for the predictors 'masfem_std' and 'gender_mf' from the negative-binomial (GLM)
    and OLS models contained in model_output.

    Returns:
      {
        "object": {
          "nb": {
            "masfem_std": {coef, se, z, pvalue, ci_low, ci_high, irr, irr_ci_low, irr_ci_high},
            "gender_mf": {...}
          },
          "ols": {
            "masfem_std": {coef, se, t, pvalue, ci_low, ci_high},
            "gender_mf": {...}
          }
        },
        "description": "Plain-language interpretation"
      }
    """
    import math
    import numpy as np

    def _safe_get_stats(model, predictors, model_type='nb'):
        """
        model_type: 'nb' for GLM negative binomial (log link), 'ols' for OLS.
        """
        results = {}
        if model is None:
            for p in predictors:
                results[p] = None
            return results

        # Check that model has expected attributes
        has_params = hasattr(model, 'params')
        has_bse = hasattr(model, 'bse')
        has_pvalues = hasattr(model, 'pvalues')
        has_conf_int = hasattr(model, 'conf_int')

        for p in predictors:
            if not (has_params and p in model.params.index):
                results[p] = None
                continue
            coef = float(model.params[p])
            se = float(model.bse[p]) if has_bse else None
            # test stat: z for GLM, t for OLS
            stat = None
            if se is not None and se != 0:
                stat = float(coef / se)
            pval = float(model.pvalues[p]) if has_pvalues else None
            ci_low, ci_high = (None, None)
            if has_conf_int:
                ci = model.conf_int().loc[p].values if hasattr(model.conf_int(), 'loc') else np.asarray(model.conf_int())[list(model.params.index).index(p)]
                ci_low, ci_high = float(ci[0]), float(ci[1])

            entry = {
                'coef': round(coef, 4),
                'se': round(se, 4) if se is not None else None,
                'stat': round(stat, 4) if stat is not None else None,
                'pvalue': round(pval, 4) if pval is not None else None,
                'ci_95%': (round(ci_low, 4) if ci_low is not None else None,
                          round(ci_high, 4) if ci_high is not None else None)
            }

            # For GLM NB with log link: also provide incidence rate ratio (exp(coef)) and CI
            if model_type == 'nb':
                try:
                    irr = math.exp(coef)
                    irr_ci_low = math.exp(ci_low) if ci_low is not None else None
                    irr_ci_high = math.exp(ci_high) if ci_high is not None else None
                    entry.update({
                        'irr': round(irr, 4),
                        'irr_95%': (round(irr_ci_low, 4) if irr_ci_low is not None else None,
                                    round(irr_ci_high, 4) if irr_ci_high is not None else None)
                    })
                except Exception:
                    entry.update({'irr': None, 'irr_95%': (None, None)})

            results[p] = entry
        return results

    predictors = ['masfem_std', 'gender_mf']
    nb_model = model_output.get('nb_model')
    ols_model = model_output.get('ols_model')

    nb_stats = _safe_get_stats(nb_model, predictors, model_type='nb')
    ols_stats = _safe_get_stats(ols_model, predictors, model_type='ols')

    # Build concise interpretation
    def interpret(p, nb_s, ols_s, alpha=0.05):
        lines = []
        # NB: fatalities
        if nb_s is None:
            lines.append(f"Fatalities model: no result for {p}.")
        else:
            coef = nb_s['coef']; pval = nb_s['pvalue']; irr = nb_s.get('irr')
            dir_text = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
            sig = (pval is not None and pval < alpha)
            lines.append(f"Fatalities (NB): coef={coef}, IRR={irr}, p={pval} -> {('significant' if sig else 'not significant')} ({dir_text}).")
        # OLS: log damage
        if ols_s is None:
            lines.append(f"Damage model: no result for {p}.")
        else:
            coef = ols_s['coef']; pval = ols_s['pvalue']
            dir_text = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
            sig = (pval is not None and pval < alpha)
            lines.append(f"Log damage (OLS): coef={coef}, p={pval} -> {('significant' if sig else 'not significant')} ({dir_text}).")
        return " ".join(lines)

    interpretation = {}
    for p in predictors:
        interpretation[p] = interpret(p, nb_stats.get(p), ols_stats.get(p))

    # Overall conclusion about the hypothesis:
    # Hypothesis: more feminine names -> less precaution -> higher fatalities or higher damage.
    # We check sign and significance for masfem_std primarily.
    masfem_nb = nb_stats.get('masfem_std')
    masfem_ols = ols_stats.get('masfem_std')
    conclusion_lines = []
    if masfem_nb is None and masfem_ols is None:
        conclusion_lines.append("No estimable results for 'masfem_std'. Cannot evaluate hypothesis.")
    else:
        # Determine whether either model shows a significant positive association (supporting hypothesis)
        nb_pos_sig = masfem_nb is not None and (masfem_nb['coef'] > 0) and (masfem_nb['pvalue'] is not None and masfem_nb['pvalue'] < 0.05)
        ols_pos_sig = masfem_ols is not None and (masfem_ols['coef'] > 0) and (masfem_ols['pvalue'] is not None and masfem_ols['pvalue'] < 0.05)
        if nb_pos_sig or ols_pos_sig:
            conclusion_lines.append("At least one model shows a statistically significant positive association between name femininity and the outcome, which would support the hypothesis.")
        else:
            # If coefficients are in opposite directions or not significant, say no evidence.
            # Report observed directions and p-values
            nb_part = ("NB coef={coef}, p={p}".format(coef=masfem_nb['coef'], p=masfem_nb['pvalue']) if masfem_nb is not None else "NB: NA")
            ols_part = ("OLS coef={coef}, p={p}".format(coef=masfem_ols['coef'], p=masfem_ols['pvalue']) if masfem_ols is not None else "OLS: NA")
            conclusion_lines.append(
                "No evidence supporting the hypothesis: neither model shows a statistically significant positive association. Observed estimates: "
                + nb_part + "; " + ols_part + "."
            )

    final_description = (
        "Extracted coefficient estimates, standard errors, test statistics, p-values, and 95% CIs for 'masfem_std' and 'gender_mf' "
        "from the negative-binomial (fatalities) and OLS (log damage) models. "
        "Interpretation: " + " ".join(conclusion_lines) + " See individual predictor summaries for details."
    )

    result_object = {
        'nb': nb_stats,
        'ols': ols_stats,
        'interpretation_by_predictor': interpretation
    }

    return {'object': result_object, 'description': final_description}