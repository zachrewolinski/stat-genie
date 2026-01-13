def extract_final_answer(model_output):
    """
    Extracts and summarizes the effect of 'HasChildren' from fitted statsmodels results
    provided in model_output (expected keys: 'logit' and 'neg_bin').

    Returns a dict with:
      - "object": a dict summarizing coefficients, p-values, 95% CIs, and exp(coef)
                  (odds ratio / incidence rate ratio) for each model and an overall
                  plain-language conclusion.
      - "description": a short explanation of what the returned numbers mean.
    """
    import numpy as np
    import pandas as pd

    summary = {}
    def summarize_model(res_obj, model_name):
        if res_obj is None:
            return {"error": f"{model_name} result is None"}
        # If the caller passed an error string instead of a model object, propagate it
        if isinstance(res_obj, str):
            return {"error": res_obj}

        try:
            params = res_obj.params            # pandas Series
            pvals = res_obj.pvalues
            ci_array = res_obj.conf_int().values  # ndarray shape (k,2)
            param_names = list(params.index)
            if 'HasChildren' not in param_names:
                return {"error": "'HasChildren' not found in model parameters"}
            idx = param_names.index('HasChildren')

            coef = float(params['HasChildren'])
            pval = float(pvals['HasChildren'])
            ci_low = float(ci_array[idx, 0])
            ci_high = float(ci_array[idx, 1])
            exp_coef = float(np.exp(coef))
            exp_ci_low = float(np.exp(ci_low))
            exp_ci_high = float(np.exp(ci_high))

            return {
                "coef": coef,
                "pvalue": pval,
                "ci_95": (ci_low, ci_high),
                "exp_coef": exp_coef,
                "exp_ci_95": (exp_ci_low, exp_ci_high)
            }
        except Exception as e:
            return {"error": f"exception while summarizing {model_name}: {str(e)}"}

    # Summarize logistic model (probability of any affair)
    logit_res = model_output.get('logit')
    summary['logit'] = summarize_model(logit_res, 'logit')

    # Summarize negative binomial model (count of affairs)
    negbin_res = model_output.get('neg_bin')
    summary['neg_bin'] = summarize_model(negbin_res, 'neg_bin')

    # Build a concise conclusion about whether having children decreases engagement in affairs
    conclusions = []
    def interpret(summ, label, kind):
        if summ is None:
            return f"{label}: no result."
        if 'error' in summ:
            return f"{label}: {summ['error']}"
        coef = summ['coef']
        p = summ['pvalue']
        expc = summ['exp_coef']
        ci = summ['ci_95']
        expci = summ['exp_ci_95']
        direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
        signif = "statistically significant" if p < 0.05 else "not statistically significant"
        if kind == 'logit':
            return (f"{label}: HasChildren coef={coef:.3f}, p={p:.3g} ({signif}); "
                    f"odds ratio={expc:.3f}, 95% CI={expci[0]:.3f}–{expci[1]:.3f}. "
                    f"Interpretation: having children is associated with a {direction} in the odds of any affair.")
        else:
            return (f"{label}: HasChildren coef={coef:.3f}, p={p:.3g} ({signif}); "
                    f"incidence rate ratio={expc:.3f}, 95% CI={expci[0]:.3f}–{expci[1]:.3f}. "
                    f"Interpretation: having children is associated with a {direction} in the expected count of affairs.")
    conclusions.append(interpret(summary['logit'], "Logistic (AnyAffair)", 'logit'))
    conclusions.append(interpret(summary['neg_bin'], "NegativeBinomial (AffairCount)", 'negbin'))

    # Combine into final short conclusion:
    # If both models show negative coef and at least one is significant, say evidence for decrease.
    def final_statement(summary):
        ok = []
        neg_signif = 0
        neg_any = 0
        pos_signif = 0
        pos_any = 0
        for k in ('logit', 'neg_bin'):
            s = summary.get(k)
            if not s or 'error' in s:
                continue
            coef = s['coef']
            p = s['pvalue']
            if coef < 0:
                neg_any += 1
                if p < 0.05:
                    neg_signif += 1
            elif coef > 0:
                pos_any += 1
                if p < 0.05:
                    pos_signif += 1
        if neg_signif >= 1 and pos_signif == 0:
            return ("Overall: Evidence that having children is associated with LOWER engagement in extramarital affairs "
                    "(at least one model shows a statistically significant negative association).")
        if neg_any >= 1 and pos_any == 0 and neg_signif == 0:
            return ("Overall: Both (or at least one) models estimate negative associations (fewer affairs for parents) "
                    "but these estimates are not statistically significant; evidence is weak.")
        if pos_signif >= 1 and neg_signif == 0:
            return ("Overall: Evidence that having children is associated with HIGHER engagement in extramarital affairs "
                    "(unexpectedly), as shown by a statistically significant positive association in at least one model.")
        if pos_any >= 1 and neg_any == 0 and pos_signif == 0:
            return ("Overall: Estimated positive associations (more affairs for parents) but not statistically significant; evidence is weak.")
        # Mixed directions or no clear signal
        return ("Overall: Mixed or inconclusive evidence across models about whether having children changes engagement in affairs; "
                "no consistent statistically significant effect.")

    final_concl = final_statement(summary)

    result_object = {
        "models": summary,
        "final_conclusion": final_concl,
        "model_level_descriptions": conclusions
    }

    description = (
        "Returned are coefficient, p-value, 95% confidence interval, and exp(coef) for 'HasChildren' "
        "from both the logistic model (odds ratio for any affair) and the negative-binomial model "
        "(incidence rate ratio for count of affairs). The 'final_conclusion' gives a plain-language "
        "summary about whether having children appears to decrease engagement in extramarital affairs "
        "based on direction and statistical significance of the estimates."
    )

    return {"object": result_object, "description": description}