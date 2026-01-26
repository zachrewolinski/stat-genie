def extract_final_answer(model_output):
    """
    Extracts the is_human effect from the provided model_output dict and returns a
    concise inference about whether modern humans have higher AMTL after controls.
    Returns a dict with keys:
      - "object": dict with numeric results (coef_logit, se, pvalue, ci_logit, odds_ratio, odds_ratio_ci, conclusion)
      - "description": short plain-language interpretation

    The function tries these in order:
      1) use model_output['is_human_inference'] if present (preferred, already computed)
      2) fall back to extracting from model_output['glm_clustered']
      3) fall back to extracting from model_output['gee']
      4) return a helpful message if none available
    """
    import numpy as np

    result = {
        "object": None,
        "description": None
    }

    # Helper to assemble output given numeric values
    def make_output(coef, se, pval, ci_lower, ci_upper):
        or_est = float(np.exp(coef))
        or_ci_lower, or_ci_upper = float(np.exp(ci_lower)), float(np.exp(ci_upper))
        conclusion = {
            "humans_higher_amtl": bool((or_est > 1.0) and (pval < 0.05)),
            "rule": "odds_ratio>1 and pvalue<0.05"
        }
        obj = {
            "coef_logit": float(coef),
            "se": float(se) if se is not None else None,
            "pvalue": float(pval),
            "ci_logit": [float(ci_lower), float(ci_upper)],
            "odds_ratio": or_est,
            "odds_ratio_ci": [or_ci_lower, or_ci_upper],
            "conclusion": conclusion
        }
        desc = (
            f"The estimated effect of is_human on AMTL (logit scale) is {obj['coef_logit']:.3f} "
            f"(SE={obj['se']:.3f}), p={obj['pvalue']:.3g}. On the odds-ratio scale the estimate is "
            f"{obj['odds_ratio']:.3f} with 95% CI [{obj['odds_ratio_ci'][0]:.3f}, {obj['odds_ratio_ci'][1]:.3f}]. "
            f"Interpretation: {'Yes — modern humans have higher AMTL after controlling for age, sex, and tooth class.' if conclusion['humans_higher_amtl'] else 'No — there is not statistically significant evidence that modern humans have higher AMTL after controls.'}"
        )
        return obj, desc

    # 1) Prefer pre-computed inference
    if isinstance(model_output, dict) and 'is_human_inference' in model_output and model_output['is_human_inference'] is not None:
        inf = model_output['is_human_inference']
        try:
            obj = {
                "coef_logit": float(inf.get('coef_logit')),
                "se": float(inf.get('se')),
                "pvalue": float(inf.get('pvalue')),
                "ci_logit": [float(inf.get('ci_logit')[0]), float(inf.get('ci_logit')[1])],
                "odds_ratio": float(inf.get('odds_ratio')),
                "odds_ratio_ci": [float(inf.get('odds_ratio_ci')[0]), float(inf.get('odds_ratio_ci')[1])],
            }
            conclusion = {"humans_higher_amtl": (obj['odds_ratio'] > 1.0 and obj['pvalue'] < 0.05),
                          "rule": "odds_ratio>1 and pvalue<0.05"}
            result['object'] = {**obj, "conclusion": conclusion}
            result['description'] = (
                f"Using the clustered GLM inference provided: coef (logit)={obj['coef_logit']:.3f}, "
                f"SE={obj['se']:.3f}, p={obj['pvalue']:.3g}. Odds ratio={obj['odds_ratio']:.3f} "
                f"(95% CI [{obj['odds_ratio_ci'][0]:.3f}, {obj['odds_ratio_ci'][1]:.3f}]). "
                + ("Conclusion: modern humans have higher AMTL after controls." if conclusion['humans_higher_amtl']
                   else "Conclusion: no statistically significant higher AMTL in modern humans after controls.")
            )
            return result
        except Exception:
            # fall through to other extraction methods if formatting unexpected
            pass

    # 2) Try extracting from glm_clustered (statsmodels GLMResults-like)
    if isinstance(model_output, dict) and 'glm_clustered' in model_output and model_output['glm_clustered'] is not None:
        glm = model_output['glm_clustered']
        coef_name = 'is_human'
        try:
            params = getattr(glm, 'params', None)
            pvalues = getattr(glm, 'pvalues', None)
            bse = getattr(glm, 'bse', None)
            conf_int = None
            try:
                conf_int = glm.conf_int()
            except Exception:
                # some wrappers use conf_int() method name
                try:
                    conf_int = glm.conf_int()
                except Exception:
                    conf_int = None

            if params is not None and coef_name in params.index:
                coef = float(params[coef_name])
                se = float(bse[coef_name]) if bse is not None and coef_name in bse.index else None
                pval = float(pvalues[coef_name]) if pvalues is not None and coef_name in pvalues.index else None
                if conf_int is not None and coef_name in conf_int.index:
                    ci_lower, ci_upper = float(conf_int.loc[coef_name][0]), float(conf_int.loc[coef_name][1])
                else:
                    # approximate using coef +/- 1.96*se if se available
                    if se is not None:
                        ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se
                    else:
                        raise ValueError("Cannot get confidence interval for is_human from GLM results.")
                obj, desc = make_output(coef, se, pval, ci_lower, ci_upper)
                result['object'] = obj
                result['description'] = "Extracted from glm_clustered: " + desc
                return result
        except Exception:
            pass

    # 3) Try extracting from gee (statsmodels GEEResults-like)
    if isinstance(model_output, dict) and 'gee' in model_output and model_output['gee'] is not None:
        gee = model_output['gee']
        coef_name = 'is_human'
        try:
            params = getattr(gee, 'params', None)
            pvalues = getattr(gee, 'pvalues', None)
            bse = getattr(gee, 'bse', None)
            conf_int = None
            try:
                conf_int = gee.conf_int()
            except Exception:
                conf_int = None

            if params is not None and coef_name in params.index:
                coef = float(params[coef_name])
                se = float(bse[coef_name]) if bse is not None and coef_name in bse.index else None
                pval = float(pvalues[coef_name]) if pvalues is not None and coef_name in pvalues.index else None
                if conf_int is not None and coef_name in conf_int.index:
                    ci_lower, ci_upper = float(conf_int.loc[coef_name][0]), float(conf_int.loc[coef_name][1])
                else:
                    if se is not None:
                        ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se
                    else:
                        raise ValueError("Cannot get confidence interval for is_human from GEE results.")
                obj, desc = make_output(coef, se, pval, ci_lower, ci_upper)
                result['object'] = obj
                result['description'] = "Extracted from GEE results: " + desc
                return result
        except Exception:
            pass

    # 4) If nothing worked, return informative fallback
    result['object'] = None
    result['description'] = (
        "Could not extract is_human effect from the provided model_output. "
        "Expected keys: 'is_human_inference', or a statsmodels results object under 'glm_clustered' or 'gee' "
        "with parameter named 'is_human'. Please provide one of these."
    )
    return result