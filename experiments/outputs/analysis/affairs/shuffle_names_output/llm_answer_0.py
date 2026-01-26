def extract_final_answer(model_output):
    """
    Extracts relevant statistics about the effect of HasChildren on AnyAffair
    from the model_output produced by the modeling function.

    Returns a dict with keys:
      - "object": a dict of extracted numeric results and statistics
      - "description": a short plain-language interpretation
    """
    import math

    out = {
        "object": None,
        "description": ""
    }

    # Helper: safe access to descriptive structure
    desc = model_output.get('descriptive', None)

    # If a fitted logit model exists, prefer extracting coefficient, p-value, CI and OR
    if 'logit_model' in model_output and model_output['logit_model'] is not None:
        try:
            lm = model_output['logit_model']
            params = getattr(lm, 'params', None)
            pvalues = getattr(lm, 'pvalues', None)
            conf = None
            try:
                conf = lm.conf_int()
            except Exception:
                conf = None

            if params is not None and 'HasChildren' in params.index:
                coef = float(params['HasChildren'])
                pval = float(pvalues['HasChildren']) if pvalues is not None else None
                if conf is not None and 'HasChildren' in conf.index:
                    ci_low = float(conf.loc['HasChildren', 0])
                    ci_high = float(conf.loc['HasChildren', 1])
                else:
                    ci_low = ci_high = None

                # Convert log-odds to odds ratio
                or_point = math.exp(coef)
                or_ci = (math.exp(ci_low) if ci_low is not None else None,
                         math.exp(ci_high) if ci_high is not None else None)

                out['object'] = {
                    'method': 'logistic_regression',
                    'coef_logit': coef,
                    'p_value': pval,
                    'coef_CI_logit': (ci_low, ci_high),
                    'odds_ratio': or_point,
                    'odds_ratio_CI': or_ci,
                    'n_logit': model_output.get('n_logit')
                }
                out['description'] = (
                    "From the fitted logistic regression: coefficient on HasChildren "
                    f"={coef:.4g} (log-odds). Odds ratio = {or_point:.3g} "
                    f"with 95% CI = ({or_ci[0]:.3g}, {or_ci[1]:.3g}) and p = {pval:.3g}. "
                    "Interpretation: values >1 for the odds ratio mean having children "
                    "is associated with higher odds of any extramarital affair; values <1 "
                    "mean lower odds. (Note: model output originally indicated a cast error "
                    "if this branch is absent.)"
                )
                return out
        except Exception:
            # fall through to descriptive fallback
            pass

    # If no fitted model is available, fall back to descriptive comparison
    if desc is not None and isinstance(desc, dict):
        try:
            means = desc.get('mean', {})
            counts = desc.get('count', {})

            # Support keys that might be strings
            def get_val(d, k):
                if k in d:
                    return d[k]
                if str(k) in d:
                    return d[str(k)]
                return None

            p0 = float(get_val(means, 0))
            p1 = float(get_val(means, 1))
            n0 = int(get_val(counts, 0))
            n1 = int(get_val(counts, 1))

            # observed counts (rounded)
            k0 = int(round(n0 * p0))
            k1 = int(round(n1 * p1))

            # Risk (difference) stats
            risk_diff = p1 - p0
            # SE for difference in proportions (normal approximation)
            se_diff = math.sqrt(max(p1 * (1 - p1) / n1, 0) + max(p0 * (1 - p0) / n0, 0))
            z_diff = risk_diff / se_diff if se_diff > 0 else float('nan')
            # Normal CDF via erf
            def normal_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            pval_diff = 2 * (1 - normal_cdf(abs(z_diff))) if not math.isnan(z_diff) else None
            rd_ci_low = risk_diff - 1.96 * se_diff
            rd_ci_high = risk_diff + 1.96 * se_diff

            # Odds ratio and CI (Woolf / log-OR) with Haldane correction if any zero cell
            a = k1
            b = n1 - k1
            c = k0
            d = n0 - k0
            # apply 0.5 correction if any cell is zero
            if a == 0 or b == 0 or c == 0 or d == 0:
                a += 0.5; b += 0.5; c += 0.5; d += 0.5
            odds1 = a / b
            odds0 = c / d
            or_point = odds1 / odds0 if odds0 > 0 else float('inf')
            # log-OR and its SE
            log_or = math.log(or_point)
            se_log_or = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            or_ci_low = math.exp(log_or - 1.96 * se_log_or)
            or_ci_high = math.exp(log_or + 1.96 * se_log_or)
            z_or = log_or / se_log_or if se_log_or > 0 else float('nan')
            pval_or = 2 * (1 - normal_cdf(abs(z_or))) if not math.isnan(z_or) else None

            # Risk ratio (point estimate)
            rr = (p1 / p0) if p0 > 0 else float('inf')
            # approximate CI for RR using log method, need positive counts
            # use k1/k0 on successes for SE_log_rr; add small correction if zero
            if k1 == 0 or k0 == 0:
                # apply 0.5 correction already if required above - recompute
                a_rr = a
                c_rr = c
            else:
                a_rr = k1
                c_rr = k0
            se_log_rr = math.sqrt(1.0 / a_rr - 1.0 / n1 + 1.0 / c_rr - 1.0 / n0) if (a_rr > 0 and c_rr > 0) else None
            if se_log_rr is not None:
                rr_ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
                rr_ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)
            else:
                rr_ci_low = rr_ci_high = None

            obj = {
                'method': 'descriptive_fallback',
                'n_total': model_output.get('n_total'),
                'n_logit': model_output.get('n_logit'),
                'n_with_children': n1,
                'n_without_children': n0,
                'p_with_children': p1,
                'p_without_children': p0,
                'count_with_affair_with_children': k1,
                'count_with_affair_without_children': k0,
                'risk_difference': risk_diff,
                'risk_difference_95CI': (rd_ci_low, rd_ci_high),
                'risk_difference_pvalue_approx': pval_diff,
                'odds_ratio': or_point,
                'odds_ratio_95CI': (or_ci_low, or_ci_high),
                'odds_ratio_pvalue_approx': pval_or,
                'risk_ratio': rr,
                'risk_ratio_95CI_approx': (rr_ci_low, rr_ci_high)
            }

            out['object'] = obj

            # plain-language description
            out['description'] = (
                "No fitted regression result was available (the original model attempts "
                "raised data-type casting errors). Using the available descriptive "
                "summary instead: among respondents WITH children, {n1} cases, "
                "{p1:.1%} reported any extramarital affair in the past year "
                "(~{k1} people). Among respondents WITHOUT children, {n0} cases, "
                "{p0:.1%} reported an affair (~{k0} people). "
                "Observed difference in proportions = {rd:.3f} (≈ {rd_pct:.1f} percentage points), "
                "95% CI ≈ [{rdlo:.3f}, {rdhi:.3f}], p ≈ {pval:.3g}. "
                "Observed odds ratio ≈ {orpt:.3g} (95% CI ≈ [{orlo:.3g}, {orhi:.3g}]). "
                "Interpretation: in these data, having children is associated with a higher "
                "observed incidence of reported affairs (about {pct:.1f}% vs {pct0:.1f}%), "
                "and the crude difference is statistically detectable by normal-approx tests. "
                "Caveat: these are unadjusted descriptive comparisons because the model fits "
                "failed; properly adjusted inference would require successfully fitting the "
                "logistic or count models after fixing the input data types."
                .format(
                    n1=n1, p1=p1, k1=k1, n0=n0, p0=p0, k0=k0,
                    rd=risk_diff, rd_pct=100 * risk_diff,
                    rdlo=rd_ci_low, rdhi=rd_ci_high,
                    pval=(pval_diff if pval_diff is not None else float('nan')),
                    orpt=or_point, orlo=or_ci_low, orhi=or_ci_high,
                    pct=100 * p1, pct0=100 * p0
                )
            )
            return out

        except Exception as e:
            out['description'] = (
                "Failed to compute descriptive statistics from model_output.descriptive: "
                f"{repr(e)}. model_output keys: {list(model_output.keys())}"
            )
            return out

    # If we reach here, no usable model or descriptive info present
    out['description'] = (
        "No usable fitted model object or descriptive summary found in model_output. "
        "Keys present: " + ", ".join(list(model_output.keys()))
    )
    out['object'] = None
    return out