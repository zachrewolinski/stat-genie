def extract_final_answer(model_output):
    """
    Extracts coefficient, p-value, confidence interval, and exponentiated effect
    (IRR or OR) for the treatment variable 'Children' from model_output.

    Returns a dict with:
      - "object": dict keyed by model name with numeric summaries
      - "description": short textual interpretation about whether having children
                       is associated with fewer extramarital affairs.
    """
    import numpy as np
    import pandas as pd

    def find_param_name(params_index, base_name='Children'):
        # Try exact match first, then case-insensitive contains
        if base_name in params_index:
            return base_name
        for n in params_index:
            if isinstance(n, str) and base_name.lower() in n.lower():
                return n
        return None

    summaries = {}
    available_models = []
    for key in ['neg_binom', 'logit_any_affair', 'zinb']:
        if key in model_output and not (isinstance(model_output[key], str) and key.endswith('_error') is False):
            res = model_output[key]
            # Some entries might be error strings; skip those
            if res is None:
                continue
            # Ensure the object looks like a fitted results with .params
            if not hasattr(res, 'params'):
                continue
            params_index = res.params.index if hasattr(res.params, 'index') else list(res.params.keys())
            pname = find_param_name(params_index, 'Children')
            if pname is None:
                summaries[key] = {'error': "Could not locate parameter name for 'Children' in model params."}
                continue

            # Extract coefficient
            try:
                coef = float(res.params[pname])
            except Exception:
                coef = float(pd.Series(res.params)[pname])

            # p-value (may not exist on some result types)
            pval = None
            try:
                if hasattr(res, 'pvalues'):
                    pval = float(res.pvalues[pname])
            except Exception:
                pval = None

            # Confidence intervals
            ci_lower = ci_upper = None
            try:
                ci = res.conf_int()
                # conf_int may be DataFrame or ndarray; use label if possible
                if isinstance(ci, (pd.DataFrame, pd.Series)):
                    ci_lower = float(ci.loc[pname][0])
                    ci_upper = float(ci.loc[pname][1])
                else:
                    # fallback: try to index by position
                    idx = list(params_index).index(pname)
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
            except Exception:
                ci_lower = ci_upper = None

            # Exponentiated coefficient (IRR for count model with log link, OR for logit)
            exp_coef = None
            exp_ci = (None, None)
            try:
                exp_coef = float(np.exp(coef))
                if (ci_lower is not None) and (ci_upper is not None):
                    exp_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
            except Exception:
                exp_coef = None

            significance = None
            if pval is not None:
                significance = bool(pval < 0.05)

            summaries[key] = {
                'param_name': pname,
                'coef': coef,
                'pvalue': pval,
                'conf_int': (ci_lower, ci_upper),
                'exp_coef': exp_coef,
                'exp_conf_int': exp_ci,
                'significant_at_0.05': significance
            }
            available_models.append(key)

    # Formulate a concise overall conclusion based on available models
    conclusion_notes = []
    sig_negative = []
    sig_positive = []
    nonsig = []
    for k, s in summaries.items():
        if 'error' in s:
            conclusion_notes.append(f"{k}: {s['error']}")
            continue
        if s['pvalue'] is None:
            nonsig.append(k)
            continue
        if s['significant_at_0.05']:
            if s['coef'] < 0:
                sig_negative.append(k)
            elif s['coef'] > 0:
                sig_positive.append(k)
            else:
                nonsig.append(k)
        else:
            nonsig.append(k)

    if len(sig_negative) > 0 and len(sig_positive) == 0:
        overall = ("Evidence that having children is associated with LOWER engagement in "
                   "extramarital affairs in the models: " + ", ".join(sig_negative) + ".")
    elif len(sig_positive) > 0 and len(sig_negative) == 0:
        overall = ("Evidence that having children is associated with HIGHER engagement in "
                   "extramarital affairs in the models: " + ", ".join(sig_positive) + ".")
    elif len(sig_negative) > 0 and len(sig_positive) > 0:
        overall = ("Mixed evidence: some models show a statistically significant negative "
                   f"association ({', '.join(sig_negative)}) while others show a significant "
                   f"positive association ({', '.join(sig_positive)}).")
    else:
        overall = ("No robust evidence that having children decreases extramarital affairs; "
                   "no model shows a statistically significant negative association at p<0.05." if len(nonsig) > 0
                   else "No models available to draw a conclusion.")

    # Build human-readable summary lines for each available model
    model_lines = []
    for k in ['neg_binom', 'logit_any_affair', 'zinb']:
        if k not in summaries:
            continue
        s = summaries[k]
        if 'error' in s:
            model_lines.append(f"{k}: {s['error']}")
            continue
        line = f"{k}: coef({s['param_name']})={s['coef']:.4f}"
        if s['pvalue'] is not None:
            line += f", p={s['pvalue']:.3g}"
        if s['exp_coef'] is not None:
            if k == 'neg_binom':
                label = "IRR"
            elif k == 'logit_any_affair':
                label = "OR"
            else:
                label = "exp(coef)"
            line += f", {label}={s['exp_coef']:.3f}"
            if s['exp_conf_int'][0] is not None:
                line += f" (95% CI {s['exp_conf_int'][0]:.3f}–{s['exp_conf_int'][1]:.3f})"
        model_lines.append(line)

    description = {
        'per_model_text': " ; ".join(model_lines) if model_lines else "No model summaries available.",
        'overall_conclusion': overall,
        'notes': ("Interpretation: For the negative binomial (count) model, exp(coef) is an incidence-rate ratio (IRR): values <1"
                  " indicate fewer expected affairs for those with children. For the logit model, exp(coef) is an odds ratio (OR).")
    }

    return {
        "object": {
            "per_model": summaries,
            "available_models": available_models,
            "final_judgment": overall
        },
        "description": description
    }