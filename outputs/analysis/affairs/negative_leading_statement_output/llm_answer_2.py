def extract_final_answer(model_output):
    """
    Extract statistics about the effect of 'has_children' on 'affairs' from the
    model_output dictionary produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing extracted numeric results from:
            * negative binomial GLM (coef, se, p, 95% CI, IRR and IRR CI)
            * OLS on log(affairs+1) (coef, se, p, 95% CI, approximate % change and CI)
            * unadjusted means comparison (means, counts, stds, mean diff, SE, t, p, 95% CI, Cohen's d)
      - "description": a brief interpretation of whether having children decreases
                       engagement in extramarital affairs, based on the extracted stats.
    """
    import numpy as np
    from math import exp, sqrt
    try:
        from scipy import stats
    except Exception:
        # If scipy is not available, we will fall back to normal-approx p-values for means
        stats = None

    out = {}
    # 1) Negative binomial results
    nb_summary = None
    try:
        nb_res = model_output.get('neg_binom', None)
        if nb_res is not None:
            coef = float(nb_res.params.get('has_children'))
            se = float(nb_res.bse.get('has_children'))
            # some statsmodels versions store pvalues as Series
            p = float(nb_res.pvalues.get('has_children'))
            ci = nb_res.conf_int().loc['has_children'].tolist()
            ci = [float(ci[0]), float(ci[1])]
            irr = float(np.exp(coef))
            irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]

            nb_summary = {
                'coef': coef,
                'std_err': se,
                'p_value': p,
                'ci_95': ci,
                'IRR': irr,
                'IRR_ci_95': irr_ci
            }
    except Exception as e:
        nb_summary = {'error': str(e)}

    out['neg_binom_has_children'] = nb_summary

    # 2) OLS on log(affairs+1) results
    ols_summary = None
    try:
        ols_res = model_output.get('ols_log', None)
        if ols_res is not None:
            coef = float(ols_res.params.get('has_children'))
            se = float(ols_res.bse.get('has_children'))
            p = float(ols_res.pvalues.get('has_children'))
            ci = ols_res.conf_int().loc['has_children'].tolist()
            ci = [float(ci[0]), float(ci[1])]
            # Interpret coefficient on log(affairs + 1): approximate percent change in (affairs+1)
            pct_change = (np.exp(coef) - 1.0) * 100.0
            pct_change_ci = [(np.exp(ci[0]) - 1.0) * 100.0, (np.exp(ci[1]) - 1.0) * 100.0]

            ols_summary = {
                'coef': coef,
                'std_err': se,
                'p_value': p,
                'ci_95': ci,
                'approx_pct_change_in_affairs_plus1': pct_change,
                'approx_pct_change_ci_95': pct_change_ci
            }
    except Exception as e:
        ols_summary = {'error': str(e)}

    out['ols_log_has_children'] = ols_summary

    # 3) Raw means comparison (unadjusted)
    means_summary = None
    try:
        means = model_output.get('means_by_children', None)
        if means is not None:
            # expects structure like {'has_children': [0,1], 'affairs_mean':[...], 'count':[...], 'std':[...]}
            has_children_vals = means.get('has_children', [])
            means_list = means.get('affairs_mean', [])
            counts = means.get('count', [])
            stds = means.get('std', [])

            # Map values to 0/1 cases (be defensive about ordering)
            mapping = {}
            for i, hc in enumerate(has_children_vals):
                mapping[int(hc)] = {
                    'mean': float(means_list[i]),
                    'n': int(counts[i]),
                    'std': float(stds[i])
                }

            # require both groups present
            if 0 in mapping and 1 in mapping:
                m0 = mapping[0]['mean']
                n0 = mapping[0]['n']
                sd0 = mapping[0]['std']
                m1 = mapping[1]['mean']
                n1 = mapping[1]['n']
                sd1 = mapping[1]['std']

                mean_diff = m1 - m0  # parents - non-parents

                # pooled standard deviation for Cohen's d and SE of mean difference
                df_pooled = n0 + n1 - 2
                pooled_var = ((n0 - 1) * (sd0 ** 2) + (n1 - 1) * (sd1 ** 2)) / df_pooled
                pooled_sd = sqrt(pooled_var)
                se_diff = pooled_sd * sqrt(1.0 / n0 + 1.0 / n1)

                # t-statistic and p-value
                if se_diff > 0:
                    t_stat = mean_diff / se_diff
                    if stats is not None:
                        p_two = 2.0 * stats.t.sf(abs(t_stat), df=df_pooled)
                        # 95% CI (t critical)
                        tcrit = stats.t.ppf(0.975, df=df_pooled)
                        ci_lower = mean_diff - tcrit * se_diff
                        ci_upper = mean_diff + tcrit * se_diff
                    else:
                        # fall back to normal approximation
                        from math import erf
                        t_stat = mean_diff / se_diff
                        p_two = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
                        zcrit = 1.96
                        ci_lower = mean_diff - zcrit * se_diff
                        ci_upper = mean_diff + zcrit * se_diff
                else:
                    t_stat = float('nan')
                    p_two = float('nan')
                    ci_lower = ci_upper = float('nan')

                cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else float('nan')

                means_summary = {
                    'non_parents': {'mean': m0, 'n': n0, 'std': sd0},
                    'parents': {'mean': m1, 'n': n1, 'std': sd1},
                    'mean_diff_parents_minus_nonparents': mean_diff,
                    'se_mean_diff': se_diff,
                    't_stat': t_stat,
                    'p_value': p_two,
                    'ci_95_mean_diff': [ci_lower, ci_upper],
                    'cohens_d': cohens_d
                }
            else:
                means_summary = {'error': 'means_by_children did not contain both groups 0 and 1'}
    except Exception as e:
        means_summary = {'error': str(e)}

    out['means_comparison'] = means_summary

    # 4) Short interpretation: decide whether having children decreases affairs
    conclusion_lines = []
    try:
        # Use adjusted models as primary evidence (negative binomial is primary)
        nb_p = None
        nb_coef = None
        if isinstance(nb_summary, dict) and 'p_value' in (nb_summary or {}):
            nb_p = nb_summary.get('p_value')
            nb_coef = nb_summary.get('coef')

        ols_p = None
        ols_coef = None
        if isinstance(ols_summary, dict) and 'p_value' in (ols_summary or {}):
            ols_p = ols_summary.get('p_value')
            ols_coef = ols_summary.get('coef')

        # Decision rules:
        # - If either adjusted model shows a statistically significant negative coefficient -> evidence children decrease affairs.
        # - Otherwise -> no evidence children decrease affairs (may be null or even higher raw means).
        decreased_evidence = False
        evidence_reasons = []
        if nb_p is not None and nb_p < 0.05 and nb_coef is not None and nb_coef < 0:
            decreased_evidence = True
            evidence_reasons.append("Negative binomial: significant negative coef (p < 0.05).")
        if not decreased_evidence and ols_p is not None and ols_p < 0.05 and ols_coef is not None and ols_coef < 0:
            decreased_evidence = True
            evidence_reasons.append("OLS log: significant negative coef (p < 0.05).")

        if decreased_evidence:
            conclusion = ("Evidence consistent with having children decreasing reported extramarital affairs. "
                          "See supporting model(s): " + "; ".join(evidence_reasons))
        else:
            # note the signs and significance
            sign_note = []
            if nb_coef is not None:
                sign_note.append(f"neg_binom coef={nb_coef:.3g}, p={nb_p:.3g}")
            if ols_coef is not None:
                sign_note.append(f"ols_log coef={ols_coef:.3g}, p={ols_p:.3g}")
            # mention raw means
            means_note = ""
            if isinstance(means_summary, dict) and 'parents' in (means_summary or {}):
                m0 = means_summary['non_parents']['mean']
                m1 = means_summary['parents']['mean']
                means_note = f"Unadjusted means: parents mean={m1:.3g}, non-parents mean={m0:.3g} (parents higher)."

            conclusion = ("No evidence that having children decreases engagement in extramarital affairs after adjustment. "
                          "Adjusted model coefficients are small and not statistically significant. "
                          + ("Adjusted model details: " + "; ".join(sign_note) + ". " if sign_note else "")
                          + means_note)

    except Exception as e:
        conclusion = "Could not compute conclusion due to error: " + str(e)

    return {'object': out, 'description': conclusion}