def extract_final_answer(model_output):
    """
    Extracts key statistics for the NameFem predictors from the model_output dictionary
    (which is expected to contain statsmodels results objects keyed as in the modeling function).
    Returns a dictionary with:
      - "object": a dictionary of extracted numeric results (coefficients, SE, p-values, 95% CIs,
                  effect-size measures: IRR for negative binomial models and percent change for
                  log-damage OLS models), plus an overall boolean 'supports_hypothesis' based
                  on the primary Deaths model.
      - "description": a concise interpretation of those statistics in the context of the task.

    Interpretation rule used to generate supports_hypothesis:
      - Primary test = nb_main_masf (Negative Binomial on Deaths).
      - Hypothesis is considered supported if the NameFem_masf coefficient is negative
        (feminine -> fewer deaths) AND its p-value < 0.05.
    """
    import numpy as np

    # Helper to extract stats for a given results object and predictor name.
    def _extract_stats(result, predictor, model_type):
        # result: statsmodels results object
        # model_type: 'nb' or 'ols' (nb -> compute IRR; ols -> percent change on logged DV)
        out = {}
        if result is None:
            return None

        # Try to safely get parameter, se, pvalue, conf_int
        try:
            coef = float(result.params[predictor])
        except Exception:
            coef = None
        try:
            se = float(result.bse[predictor])
        except Exception:
            se = None
        try:
            pval = float(result.pvalues[predictor])
        except Exception:
            pval = None
        # confidence interval: result.conf_int() returns DataFrame or array
        try:
            ci = result.conf_int().loc[predictor].astype(float)
            ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])
        except Exception:
            # fallback: try array indexing (if no index labels)
            try:
                ci_arr = result.conf_int()
                idx = list(result.params.index).index(predictor)
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                ci_low, ci_high = None, None

        out['coef'] = coef
        out['se'] = se
        out['p_value'] = pval
        out['95CI'] = (ci_low, ci_high)

        if coef is not None:
            if model_type == 'nb':
                # incidence rate ratio
                irr = float(np.exp(coef))
                irr_ci_low = float(np.exp(ci_low)) if (ci_low is not None) else None
                irr_ci_high = float(np.exp(ci_high)) if (ci_high is not None) else None
                out['effect_size'] = {'IRR': irr, 'IRR_95CI': (irr_ci_low, irr_ci_high)}
                out['interpretation'] = (
                    "Negative coefficient -> IRR < 1 means higher femininity associated with fewer deaths."
                    if coef < 0 else
                    "Positive coefficient -> IRR > 1 means higher femininity associated with more deaths."
                )
            elif model_type == 'ols':
                # DV is log_ndam15. Transform to percent change approximation: (exp(beta)-1)*100
                pct = (np.exp(coef) - 1.0) * 100.0
                pct_ci_low = (np.exp(ci_low) - 1.0) * 100.0 if (ci_low is not None) else None
                pct_ci_high = (np.exp(ci_high) - 1.0) * 100.0 if (ci_high is not None) else None
                out['effect_size'] = {'percent_change': pct, 'percent_change_95CI': (pct_ci_low, pct_ci_high)}
                out['interpretation'] = (
                    "Negative coefficient -> negative percent change means higher femininity associated with lower logged damage."
                    if coef < 0 else
                    "Positive coefficient -> positive percent change means higher femininity associated with greater logged damage."
                )
        return out

    # Keys expected in model_output
    keys_map = {
        'nb_main_masf': ('NameFem_masf', 'nb'),
        'ols_damage_main_masf': ('NameFem_masf', 'ols'),
        'nb_mturk': ('NameFem_mturk', 'nb'),
        'ols_damage_mturk': ('NameFem_mturk', 'ols'),
        'nb_bin': ('NameFem_bin', 'nb'),
        'ols_damage_bin': ('NameFem_bin', 'ols')
    }

    extracted = {}
    for key, (pred, mtype) in keys_map.items():
        res = model_output.get(key)
        if res is None:
            extracted[key] = None
        else:
            try:
                extracted[key] = _extract_stats(res, pred, mtype)
            except Exception as e:
                extracted[key] = {'error': str(e)}

    # Formulate primary conclusion using the primary negative binomial test
    primary = extracted.get('nb_main_masf')
    supports = None
    conclusion_text = "Insufficient information to form a conclusion for the primary test."
    if primary and 'coef' in primary and primary['coef'] is not None and primary['p_value'] is not None:
        coef = primary['coef']
        pval = primary['p_value']
        if coef < 0 and pval < 0.05:
            supports = True
            conclusion_text = (
                "Primary negative-binomial result: coefficient is negative and statistically significant "
                f"(coef={coef:.4g}, p={pval:.4g}). This provides evidence that more feminine hurricane names "
                "are associated with fewer deaths (consistent with the hypothesis)."
            )
        else:
            supports = False
            conclusion_text = (
                "Primary negative-binomial result does NOT support the hypothesis: "
                f"coef={coef:.4g}, p={pval:.4g}. "
                "Either the coefficient is not negative or it is not statistically significant at p<0.05."
            )

    # Short summary for the OLS log damage main result
    ols_main = extracted.get('ols_damage_main_masf')
    if ols_main and 'coef' in ols_main and ols_main['coef'] is not None:
        coef = ols_main['coef']
        pval = ols_main['p_value']
        pct = ols_main['effect_size']['percent_change'] if ols_main.get('effect_size') else None
        conclusion_text += " OLS (log damage) main result: "
        conclusion_text += (
            f"coef={coef:.4g}, p={pval:.4g}, implied percent-change={pct:.3g}%."
            if (coef is not None and pval is not None and pct is not None)
            else "result available but could not be fully summarized."
        )

    result_object = {
        'models': extracted,
        'supports_hypothesis': supports,
        'primary_conclusion': conclusion_text
    }

    description = (
        "Returned object contains for each specification: coefficient, SE, p-value, 95% CI, and an interpretable "
        "effect-size (IRR for negative binomial on Deaths; percent change for OLS on logged damage). "
        "The key decision rule used: the primary hypothesis (feminine names -> fewer deaths) is considered supported "
        "only if the NameFem_masf coefficient in the primary negative-binomial model is negative AND statistically significant (p < 0.05). "
        "Inspect result_object['models'] for full numbers; result_object['primary_conclusion'] gives a concise verdict."
    )

    return {"object": result_object, "description": description}