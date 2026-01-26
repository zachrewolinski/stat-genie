def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of having children on extramarital affairs
    from the modeling output and produces a concise conclusion.

    Returns a dict with keys:
      - "object": a dictionary with extracted numeric results and a programmatic conclusion
      - "description": a short human-readable explanation of what the numbers mean
    """
    out = {}
    # Helper to safely retrieve sub-dicts
    logit = model_output.get('logit_OR_children')
    nb = model_output.get('nb_children_IRR')
    n_total = model_output.get('n_total')
    n_any = model_output.get('n_any_affair')
    n_children = model_output.get('n_children_yes')

    results = {}

    # Extract logistic results (any affair)
    if logit is not None:
        try:
            or_val = float(logit.get('odds_ratio', None))
            or_ci = (float(logit.get('95%_CI_lower', None)), float(logit.get('95%_CI_upper', None)))
            or_p = float(logit.get('pvalue', None))
            results['logistic_any_affair'] = {
                'odds_ratio': or_val,
                '95%_CI': or_ci,
                'pvalue': or_p
            }
        except Exception:
            results['logistic_any_affair'] = logit  # return raw if unexpected format
    else:
        results['logistic_any_affair'] = None

    # Extract count-model results (IRR among those with >0 affairs)
    if nb is not None:
        try:
            irr = float(nb.get('IRR', None))
            irr_ci = (float(nb.get('95%_CI_lower', None)), float(nb.get('95%_CI_upper', None)))
            irr_p = float(nb.get('pvalue', None))
            results['count_IRR_positive_affairs'] = {
                'IRR': irr,
                '95%_CI': irr_ci,
                'pvalue': irr_p
            }
        except Exception:
            results['count_IRR_positive_affairs'] = nb
    else:
        results['count_IRR_positive_affairs'] = None

    # Build a concise conclusion based on p-values and direction
    conclusions = []
    sig_level = 0.05

    # Interpret logistic
    logi = results.get('logistic_any_affair')
    if logi:
        or_val = logi['odds_ratio']
        p = logi['pvalue']
        if p < sig_level:
            if or_val < 1:
                conclusions.append("Logistic: statistically significant decrease in odds of any affair associated with having children.")
            else:
                conclusions.append("Logistic: statistically significant increase in odds of any affair associated with having children.")
        else:
            conclusions.append("Logistic: effect not statistically significant (no strong evidence that having children changes odds of any affair).")
    else:
        conclusions.append("Logistic: results unavailable.")

    # Interpret count model
    cnt = results.get('count_IRR_positive_affairs')
    if cnt:
        irr = cnt['IRR']
        p = cnt['pvalue']
        if p < sig_level:
            if irr < 1:
                conclusions.append("Count model: statistically significant lower rate (fewer affairs among those who have had affairs) associated with having children.")
            else:
                conclusions.append("Count model: statistically significant higher rate associated with having children.")
        else:
            conclusions.append("Count model: effect not statistically significant (no strong evidence of a change in number of affairs among those with affairs).")
    else:
        conclusions.append("Count model: results unavailable or not fitted (too few positive-affair observations).")

    # Final overall assessment: require consistent statistically significant evidence of decrease to answer "Yes"
    overall = "Inconclusive"
    # If both models are present and both show significant decreases -> conclude decrease.
    if (results.get('logistic_any_affair') and results.get('count_IRR_positive_affairs')):
        logi_p = results['logistic_any_affair']['pvalue']
        cnt_p = results['count_IRR_positive_affairs']['pvalue']
        logi_dir = results['logistic_any_affair']['odds_ratio']
        cnt_dir = results['count_IRR_positive_affairs']['IRR']
        if (logi_p < sig_level and logi_dir < 1) and (cnt_p < sig_level and cnt_dir < 1):
            overall = "Yes, having children decreases engagement (statistically significant in both models)."
        else:
            overall = "No strong evidence that having children decreases engagement in extramarital affairs (effects not consistently statistically significant)."
    else:
        # If only logistic present and significant decrease:
        if results.get('logistic_any_affair') and results['logistic_any_affair']['pvalue'] < sig_level and results['logistic_any_affair']['odds_ratio'] < 1:
            overall = "Some evidence that having children decreases odds of any affair (statistically significant in logistic model only)."
        # If only count model present and significant decrease:
        elif results.get('count_IRR_positive_affairs') and results['count_IRR_positive_affairs']['pvalue'] < sig_level and results['count_IRR_positive_affairs']['IRR'] < 1:
            overall = "Some evidence that having children decreases number of affairs among those who have had affairs (statistically significant in count model only)."
        else:
            overall = "No strong evidence that having children decreases engagement in extramarital affairs."

    # Assemble final object
    out['object'] = {
        'extracted_results': results,
        'sample_sizes': {
            'n_total': int(n_total) if n_total is not None else None,
            'n_any_affair': int(n_any) if n_any is not None else None,
            'n_children_yes': int(n_children) if n_children is not None else None
        },
        'conclusion': overall
    }

    # Short human-readable description
    desc_parts = []
    if results.get('logistic_any_affair'):
        l = results['logistic_any_affair']
        desc_parts.append(f"Logistic (any affair): OR={l['odds_ratio']:.3f}, 95% CI=({l['95%_CI'][0]:.3f}, {l['95%_CI'][1]:.3f}), p={l['pvalue']:.3f}.")
    else:
        desc_parts.append("Logistic (any affair): not available.")

    if results.get('count_IRR_positive_affairs'):
        c = results['count_IRR_positive_affairs']
        desc_parts.append(f"Count model (positive affairs): IRR={c['IRR']:.3f}, 95% CI=({c['95%_CI'][0]:.3f}, {c['95%_CI'][1]:.3f}), p={c['pvalue']:.3f}.")
    else:
        desc_parts.append("Count model (positive affairs): not available or not fitted.")

    desc_parts.append(f"Sample: n_total={out['object']['sample_sizes']['n_total']}, n_any_affair={out['object']['sample_sizes']['n_any_affair']}, n_children_yes={out['object']['sample_sizes']['n_children_yes']}.")
    desc_parts.append("Overall: " + overall)

    out['description'] = " ".join(desc_parts)
    return out

# Example usage:
# final = extract_final_answer(model_output)
# print(final['description'])