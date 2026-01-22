def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of 'Children' on number of affairs
    from the provided model_output dictionary and returns a concise interpretation.

    Returns a dict with:
      - "object": a dict of numeric results (coef, p-values, IRR, CI, overdispersion, percent change)
      - "description": a short plain-language interpretation answering whether having children
                       decreases engagement in extramarital affairs (and if the effect is
                       statistically significant).
    """
    # Prepare default values
    result = {
        'children_coef_poisson': None,
        'children_pval_poisson': None,
        'children_coef_nb': None,
        'children_pval_nb': None,
        'children_IRR_nb': None,
        'children_IRR_CI_lower_nb': None,
        'children_IRR_CI_upper_nb': None,
        'percent_change_rate_nb': None,
        'overdispersion_poisson_deviance_per_df': None
    }

    try:
        # children_summary if present (from the provided model function)
        cs = model_output.get('children_summary', {}) if isinstance(model_output, dict) else {}
        if cs:
            result['children_coef_poisson'] = float(cs.get('children_coef_poisson')) if cs.get('children_coef_poisson') is not None else None
            result['children_pval_poisson'] = float(cs.get('children_pval_poisson')) if cs.get('children_pval_poisson') is not None else None
            result['children_coef_nb'] = float(cs.get('children_coef_nb')) if cs.get('children_coef_nb') is not None else None
            result['children_pval_nb'] = float(cs.get('children_pval_nb')) if cs.get('children_pval_nb') is not None else None

        # overdispersion number if present
        od = model_output.get('overdispersion', None) if isinstance(model_output, dict) else None
        if od is not None:
            result['overdispersion_poisson_deviance_per_df'] = float(od)

        # Try to get IRR and CI from 'nb_irrs' DataFrame-like object
        nb_irrs = model_output.get('nb_irrs', None) if isinstance(model_output, dict) else None
        if nb_irrs is not None:
            # nb_irrs may be a pandas DataFrame; handle using .loc or dict-like access
            try:
                # prefer DataFrame .loc
                irr_row = nb_irrs.loc['Children']
                result['children_IRR_nb'] = float(irr_row.get('IRR', irr_row[1]))
                result['children_IRR_CI_lower_nb'] = float(irr_row.get('IRR_lower', irr_row[2]))
                result['children_IRR_CI_upper_nb'] = float(irr_row.get('IRR_upper', irr_row[3]))
            except Exception:
                # fallback: try dict-style
                try:
                    row = nb_irrs['Children']
                    result['children_IRR_nb'] = float(row.get('IRR'))
                    result['children_IRR_CI_lower_nb'] = float(row.get('IRR_lower'))
                    result['children_IRR_CI_upper_nb'] = float(row.get('IRR_upper'))
                except Exception:
                    # give up silently; values remain None
                    pass

        # Compute percent change in expected rate: (IRR - 1) * 100
        if result['children_IRR_nb'] is not None:
            result['percent_change_rate_nb'] = float((result['children_IRR_nb'] - 1.0) * 100.0)

    except Exception as e:
        # If something unexpected happens, include the error message in description below
        err = str(e)
    # Build the plain-language description / conclusion
    # Use available numbers to form the conclusion
    desc_parts = []

    # Overdispersion note
    od_val = result['overdispersion_poisson_deviance_per_df']
    if od_val is not None:
        desc_parts.append(f"Poisson overdispersion (deviance/df) = {od_val:.3f} (>1 indicates overdispersion).")

    # NB IRR summary
    irr = result['children_IRR_nb']
    ci_l = result['children_IRR_CI_lower_nb']
    ci_u = result['children_IRR_CI_upper_nb']
    p_nb = result['children_pval_nb']
    coef_nb = result['children_coef_nb']

    if irr is not None and ci_l is not None and ci_u is not None:
        pct = result['percent_change_rate_nb']
        desc_parts.append(
            f"Negative binomial model: IRR for Children = {irr:.3f} "
            f"(95% CI {ci_l:.3f} to {ci_u:.3f}), corresponding to a {pct:.1f}% change in the expected rate of affairs."
        )
        # Statistical significance
        if p_nb is not None:
            if p_nb < 0.05:
                sig_text = f"The effect is statistically significant (p = {p_nb:.3f})."
            else:
                sig_text = f"The effect is not statistically significant (p = {p_nb:.3f})."
            desc_parts.append(sig_text)
        else:
            desc_parts.append("P-value for the NB coefficient not available.")
    else:
        # fallback to coefficient and p-value if IRR missing
        if coef_nb is not None:
            desc_parts.append(f"Negative binomial coefficient for Children = {coef_nb:.4f}.")
            if p_nb is not None:
                if p_nb < 0.05:
                    desc_parts.append(f"The effect is statistically significant (p = {p_nb:.3f}).")
                else:
                    desc_parts.append(f"The effect is not statistically significant (p = {p_nb:.3f}).")
        else:
            desc_parts.append("Insufficient information to summarize the Children effect from the NB model.")

    # Final verdict answering whether having children decreases engagement in extramarital affairs
    final_verdict = ""
    if irr is not None:
        if (irr < 1.0) and (p_nb is not None) and (p_nb < 0.05):
            final_verdict = "Conclusion: Having children is associated with a statistically significant decrease in the expected rate of extramarital affairs."
        elif (irr < 1.0) and (p_nb is not None) and (p_nb >= 0.05):
            final_verdict = "Conclusion: Having children is associated with a small decrease in the expected rate of affairs (IRR < 1), but this decrease is not statistically significant — no strong evidence that children reduce affairs."
        elif (irr >= 1.0) and (p_nb is not None) and (p_nb < 0.05):
            final_verdict = "Conclusion: Having children is associated with a statistically significant increase in the expected rate of extramarital affairs."
        else:
            final_verdict = "Conclusion: No statistically significant evidence that having children changes the expected rate of extramarital affairs."
        desc_parts.append(final_verdict)
    else:
        desc_parts.append("Conclusion: Unable to determine a clear effect from the available model output.")

    description = " ".join(desc_parts)

    return {
        "object": result,
        "description": description
    }