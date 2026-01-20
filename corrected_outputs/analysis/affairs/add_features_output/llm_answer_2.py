def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'children_binary' on 'affairs' from:
      - OLS results (model_output['ols'])
      - Zero-inflated count results (model_output['zinb_or_zip'])
    Returns a dict with:
      - "object": dict of extracted numeric results (coefficients, SEs, p-values, CIs, IRRs)
      - "description": short textual interpretation about whether having children decreases affairs
    
    The function is written to be robust to parameter naming conventions used by statsmodels'
    zero-inflated models (it uses model.exog_names and model.exog_infl_names when available).
    """
    import numpy as np
    import pandas as pd

    out = {
        'ols': None,
        'zinb_or_zip': None,
        'conclusion': None
    }

    def safe_get_from_res(res, name):
        """Try to get series elements robustly (handles pandas Series or numpy arrays)."""
        try:
            # If params/pvalues/bse are pandas Series, this works
            return res[name]
        except Exception:
            # Fall back: if res is array-like, can't get by name
            return None

    # 1) OLS extraction
    ols = model_output.get('ols')
    if ols is not None:
        try:
            # Coefficient, SE, p-value, 95% CI
            coef = float(ols.params['children_binary'])
            se = float(ols.bse['children_binary'])
            pval = float(ols.pvalues['children_binary'])
            ci_lower, ci_upper = ols.conf_int().loc['children_binary'].tolist()
            out['ols'] = {
                'coef': coef,
                'se': se,
                'pvalue': pval,
                '95%_ci': [float(ci_lower), float(ci_upper)],
                'interpretation': ("OLS coef is the average difference in number of reported affairs "
                                   "associated with having children (children_binary=1 vs 0).")
            }
        except Exception:
            # Attempt name-robust search if exact key not present
            try:
                # find parameter name that contains 'children_binary'
                param_names = list(ols.params.index)
                candidates = [n for n in param_names if 'children_binary' in n]
                if candidates:
                    name = candidates[0]
                    coef = float(ols.params[name])
                    se = float(ols.bse[name])
                    pval = float(ols.pvalues[name])
                    ci_lower, ci_upper = ols.conf_int().loc[name].tolist()
                    out['ols'] = {
                        'coef': coef,
                        'se': se,
                        'pvalue': pval,
                        '95%_ci': [float(ci_lower), float(ci_upper)],
                        'param_name_used': name,
                        'interpretation': ("OLS coef is the average difference in number of reported affairs "
                                           "associated with having children (children_binary=1 vs 0).")
                    }
                else:
                    out['ols'] = {'error': "children_binary not found in OLS parameters"}
            except Exception as e:
                out['ols'] = {'error': f"Failed to extract OLS results: {e}"}
    else:
        out['ols'] = {'error': 'OLS result missing'}

    # 2) Zero-inflated count extraction (ZINB or ZIP)
    zinb = model_output.get('zinb_or_zip')
    if zinb is not None:
        try:
            model = getattr(zinb, 'model', None)
            params = zinb.params  # typically a pandas Series
            pvalues = getattr(zinb, 'pvalues', None)
            bse = getattr(zinb, 'bse', None)
            ci = zinb.conf_int()

            # Identify count-side and inflation-side parameter names
            count_names = []
            infl_names = []
            if model is not None:
                # model.exog_names and model.exog_infl_names are reliable
                try:
                    count_names = list(model.exog_names)
                except Exception:
                    count_names = []
                try:
                    infl_names = list(model.exog_infl_names)
                except Exception:
                    infl_names = []
            else:
                # fallback: inspect param index names
                all_names = list(params.index)
                # heuristic: inflation names often start with 'inflate_' or 'inflate.' or 'infl_'
                infl_prefixes = ('inflate_', 'inflate.', 'infl_')
                for n in all_names:
                    if any(n.startswith(p) for p in infl_prefixes) or ('inflate' in n and 'children_binary' in n):
                        infl_names.append(n)
                    else:
                        count_names.append(n)

            # find the exact parameter name for children_binary on each side
            def find_param_in(names_list, target='children_binary'):
                # prefer exact match, else substring match
                for n in names_list:
                    if n == target:
                        return n
                for n in names_list:
                    if target in n:
                        return n
                # not found
                return None

            count_param = find_param_in(count_names, 'children_binary')
            infl_param = find_param_in(infl_names, 'children_binary')

            zinb_res_dict = {}

            # Extract count-side results
            if count_param is not None and count_param in params.index:
                c_coef = float(params[count_param])
                c_se = float(bse[count_param]) if (bse is not None and count_param in bse.index) else None
                c_p = float(pvalues[count_param]) if (pvalues is not None and count_param in pvalues.index) else None
                try:
                    c_ci_low, c_ci_high = ci.loc[count_param].tolist()
                except Exception:
                    c_ci_low = c_ci_high = None
                # Incident Rate Ratio (IRR) and CI
                irr = float(np.exp(c_coef))
                irr_ci = [None, None]
                if (c_ci_low is not None) and (c_ci_high is not None):
                    irr_ci = [float(np.exp(c_ci_low)), float(np.exp(c_ci_high))]
                zinb_res_dict['count'] = {
                    'param_name': count_param,
                    'coef': c_coef,
                    'se': c_se,
                    'pvalue': c_p,
                    '95%_ci_coef': [c_ci_low, c_ci_high],
                    'IRR': irr,
                    '95%_ci_IRR': irr_ci,
                    'interpretation': ("Count model coef is the effect (log scale) of children on the expected "
                                       "count of affairs; IRR < 1 means fewer expected affairs when children present.")
                }
            else:
                zinb_res_dict['count'] = {'error': 'children_binary not found on count side'}

            # Extract inflation-side results (zero-inflation logistic model)
            if infl_param is not None and infl_param in params.index:
                i_coef = float(params[infl_param])
                i_se = float(bse[infl_param]) if (bse is not None and infl_param in bse.index) else None
                i_p = float(pvalues[infl_param]) if (pvalues is not None and infl_param in pvalues.index) else None
                try:
                    i_ci_low, i_ci_high = ci.loc[infl_param].tolist()
                except Exception:
                    i_ci_low = i_ci_high = None
                # For inflation side, positive coef means greater log-odds of being in the always-zero (inflated) group
                zinb_res_dict['inflation'] = {
                    'param_name': infl_param,
                    'coef': i_coef,
                    'se': i_se,
                    'pvalue': i_p,
                    '95%_ci_coef': [i_ci_low, i_ci_high],
                    'odds_ratio': float(np.exp(i_coef)),
                    'interpretation': ("Inflation-side coef is effect of children on log-odds of being in the "
                                       "excess-zero group (i.e., having structural zero probability). "
                                       "Positive -> more likely to be an always-zero (no affairs).")
                }
            else:
                zinb_res_dict['inflation'] = {'error': 'children_binary not found on inflation side'}

            out['zinb_or_zip'] = zinb_res_dict

            # Formulate a combined conclusion using significance (alpha=0.05) if p-values available
            conclusions = []
            evidence_count = False
            evidence_infl = False
            try:
                # Count-side: negative coef & p < .05 => fewer affairs
                cnt = zinb_res_dict.get('count', {})
                if 'coef' in cnt and cnt['pvalue'] is not None:
                    if (cnt['coef'] < 0) and (cnt['pvalue'] < 0.05):
                        conclusions.append("Count-side: having children is significantly associated with fewer reported affairs (IRR < 1).")
                        evidence_count = True
                    elif cnt['pvalue'] < 0.05:
                        conclusions.append("Count-side: having children is significantly associated with a change in expected number of affairs (coef has opposite sign).")
                    else:
                        conclusions.append("Count-side: no statistically significant association between having children and expected number of affairs (p >= 0.05).")
                else:
                    conclusions.append("Count-side: insufficient information to assess statistical significance.")
                # Inflation-side: positive coef & p < .05 => more likely structural zero -> fewer affairs
                inf = zinb_res_dict.get('inflation', {})
                if 'coef' in inf and inf['pvalue'] is not None:
                    if (inf['coef'] > 0) and (inf['pvalue'] < 0.05):
                        conclusions.append("Inflation-side: having children significantly increases the odds of being in the 'always zero' group (less likely to have any affairs).")
                        evidence_infl = True
                    elif inf['pvalue'] < 0.05:
                        conclusions.append("Inflation-side: having children significantly affects odds of being in the always-zero group (coef has opposite sign).")
                    else:
                        conclusions.append("Inflation-side: no statistically significant association on the inflation (structural-zero) probability (p >= 0.05).")
                else:
                    conclusions.append("Inflation-side: insufficient information to assess statistical significance.")
            except Exception:
                conclusions.append("Unable to automatically determine significance-based conclusion.")

            # Synthesize final short conclusion
            if evidence_count or evidence_infl:
                out['conclusion'] = ("Overall: Evidence suggests having children is associated with reduced engagement in extramarital affairs. "
                                     "See count-side (effect on expected count) and inflation-side (effect on probability of being an assured zero) results for details.")
            else:
                out['conclusion'] = ("Overall: No strong evidence that having children decreases engagement in extramarital affairs based on the zero-inflated model (no significant count or inflation effects). "
                                     "Check p-values and confidence intervals in the returned object for details.")

            # Compose description summarizing what is returned
            description_lines = [
                "This output contains:",
                "- OLS: coefficient, SE, p-value, and 95% CI interpreting the mean-difference in number of affairs.",
                "- ZINB/ZIP count-side: log-coefficient, SE, p-value, 95% CI, IRR (exp(coef)) and IRR 95% CI. IRR < 1 indicates fewer expected affairs when children are present.",
                "- ZINB/ZIP inflation-side: logistic coefficient, SE, p-value, 95% CI, and odds ratio. Positive inflation coef indicates higher odds of being a structural zero (no affairs).",
                "",
                "Final synthesized conclusion (based on statistical significance at alpha=0.05) is in out['conclusion']."
            ]
            description = "\n".join(description_lines)

            return {
                "object": out,
                "description": description
            }

        except Exception as e:
            return {
                "object": {"error": f"Failed to extract from zinb_or_zip result: {e}"},
                "description": "Extraction from zero-inflated model failed."
            }
    else:
        return {
            "object": out,
            "description": "No zero-inflated model result provided; OLS extraction attempted above."
        }